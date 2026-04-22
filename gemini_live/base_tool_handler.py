"""
Tool handler base class for the Gemini Live Framework.

Users subclass BaseToolHandler and define async methods whose names match
their Gemini function declarations.  The framework dispatches tool calls
to these methods automatically via getattr.

The handler is fully self-contained: it owns dedup, blocking/non-blocking
scheduling, and background task management.  It emits each ``ToolHandlerResult``
through a send-tool-result callback wired by the Orchestrator — the
Orchestrator performs the final cancellation check and forwards the result
to Gemini.

Every tool method must be decorated with ``@tool`` / ``@tool(...)``.  The
decorator acts as both the config attachment point and the allowlist the
dispatcher uses to decide which methods Gemini is allowed to invoke —
undecorated methods (including framework internals) cannot be called as
tools.

Example
-------
    from gemini_live.base_tool_handler import BaseToolHandler, tool

    class MyTools(BaseToolHandler):

        @tool()
        async def get_clock(self) -> dict:
            return {"time": datetime.now().isoformat()}

        @tool(blocking=False, execution_delay=3.0, interim_message="Looking it up…")
        async def fetch_knowledge(self, query: str) -> dict:
            return {"answer": await search(query)}

        async def on_complete(self, tool_call, result):
            if result.get("end_call"):
                await self.state.schedule_hangup()
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .models import ToolCallData

logger = logging.getLogger(__name__)

DEDUP_COOLDOWN_SECONDS = 6.0


@dataclass
class ToolConfig:
    """Configuration for a single tool's execution behaviour."""
    blocking: bool = True
    execution_delay: float = 2.0
    interim_message: str = ""


def tool(
    blocking: bool = True,
    execution_delay: float = 2.0,
    interim_message: str = "",
) -> Callable:
    """Decorator that marks a method as a Gemini-callable tool and attaches
    its ``ToolConfig``.

    Only decorated methods appear in the handler's dispatch registry — this
    is what prevents Gemini from invoking framework internals or consumer
    helpers by hallucinating their names.
    """
    config = ToolConfig(
        blocking=blocking,
        execution_delay=execution_delay,
        interim_message=interim_message,
    )

    def decorator(fn: Callable) -> Callable:
        fn._tool_config = config  # type: ignore[attr-defined]
        return fn

    return decorator


class ToolResponseAction(Enum):
    """Action the Orchestrator should take for a tool result."""
    SEND_RESPONSE = "send_response"
    SEND_INTERIM = "send_interim"
    SEND_CONTEXT = "send_context"


@dataclass
class ToolHandlerResult:
    """Structured result emitted to the Orchestrator for forwarding to Gemini."""
    action: ToolResponseAction
    tool_id: str
    tool_name: str
    result: Optional[Dict[str, Any]] = None
    interim_message: str = ""


class BaseToolHandler:
    """Base class for tool execution in the Gemini Live Framework.

    Subclass this, define ``async def <tool_name>(self, ...)`` methods for
    each Gemini function declaration, and pass an instance to the
    Orchestrator.

    The handler owns the full tool lifecycle: dedup → config lookup →
    blocking / non-blocking dispatch → ``_send_tool_result``. Each
    ``ToolHandlerResult`` flows through the ``send_tool_result`` callback
    the Orchestrator wires at startup; the Orchestrator performs the final
    ``_cancelled_ids`` check immediately before sending to Gemini, closing
    the race where a cancel arrives after the tool finishes but before its
    result reaches the wire.
    """

    def __init__(self):
        self._send_tool_result: Optional[
            Callable[[ToolHandlerResult], Awaitable[None]]
        ] = None
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        self._processed_tool_ids: Dict[str, float] = {}
        self._in_flight_hashes: Dict[str, float] = {}
        self._cancelled_ids: Dict[str, float] = {}

    def set_send_tool_result(
        self,
        send_tool_result: Callable[[ToolHandlerResult], Awaitable[None]],
    ) -> None:
        """Register the Orchestrator's result sink.

        Called once during Orchestrator startup. Every ``ToolHandlerResult``
        the handler produces is forwarded to this callback, which performs
        the final cancel-check and sends to Gemini.
        """
        self._send_tool_result = send_tool_result

    # --- User API (override in subclass) ---------------------------------

    async def on_complete(
        self, tool_call: ToolCallData, result: Dict[str, Any]
    ) -> None:
        """Called after a tool finishes execution.

        Override for side-effects such as ending a call or updating state.
        """
        pass

    async def on_cancelled(self, tool_ids: List[str]) -> None:
        """Called when Gemini cancels in-flight tool calls.

        Override to abort background work or clean up resources.
        """
        pass

    # --- Framework interface (called by the Orchestrator) ----------------

    async def handle_tool_call(self, tool_call: ToolCallData) -> None:
        """Process an incoming tool call (dedup → dispatch → emit result)."""
        if self._is_duplicate(tool_call):
            return

        config = self._get_tool_config(tool_call.name)
        if config is None:
            logger.warning(
                f"[ToolHandler] Rejecting tool={tool_call.name} id={tool_call.id} "
                "reason=not_registered_with_@tool"
            )
            await self._send_tool_result(ToolHandlerResult(
                action=ToolResponseAction.SEND_RESPONSE,
                tool_id=tool_call.id,
                tool_name=tool_call.name,
                result={"success": False, "error": f"Unknown tool: {tool_call.name}"},
            ))
            self._finish_hash(self._compute_hash(tool_call.name, tool_call.args))
            return

        if config.blocking:
            await self._handle_blocking(tool_call)
        else:
            await self._handle_non_blocking(tool_call, config)

    async def handle_cancellation(self, tool_ids: List[str]) -> None:
        """Cancel pending (non-blocking) tool tasks and notify subclass.

        Also records the ids so any result that has already finished execution
        but is still en-route to the Orchestrator (or about to be emitted) is
        suppressed instead of leaking into the next turn as stale context.
        The Orchestrator performs a final re-check on its side right before
        sending to Gemini, closing the race in the callback path.
        """
        cancelled = 0
        now = time.monotonic()
        for tool_id in tool_ids:
            self._cancelled_ids[tool_id] = now
            task = self._pending_tasks.get(tool_id)
            if task and not task.done():
                task.cancel()
                cancelled += 1
        if cancelled:
            logger.info(f"[ToolHandler] Cancelled {cancelled}/{len(tool_ids)} pending tasks")
        await self.on_cancelled(tool_ids)

    async def cleanup(self) -> None:
        """Cancel all pending tasks and clear dedup state."""
        for task in self._pending_tasks.values():
            if not task.done():
                task.cancel()
        self._pending_tasks.clear()
        self._processed_tool_ids.clear()
        self._in_flight_hashes.clear()
        self._cancelled_ids.clear()

    # --- Internals -------------------------------------------------------

    def _get_tool_config(self, tool_name: str) -> Optional[ToolConfig]:
        """Return the ``@tool`` decorator config, or ``None`` if the method is
        missing, non-callable, or not decorated.
        """
        method = getattr(self, tool_name, None)
        if method is None or not callable(method):
            return None
        return getattr(method, "_tool_config", None)

    async def _handle_blocking(self, tool_call: ToolCallData) -> None:
        tool_hash = self._compute_hash(tool_call.name, tool_call.args)
        logger.info(f"[ToolHandler] Executing tool={tool_call.name} id={tool_call.id} mode=blocking")
        try:
            result = await self._execute(tool_call)
        except Exception as e:
            logger.error(
                "[ToolHandler] Execution failed tool=%s mode=blocking: %s",
                tool_call.name, e, exc_info=True,
            )
            result = {"success": False, "error": str(e)}

        if tool_call.id in self._cancelled_ids:
            logger.info(
                f"[ToolHandler] Suppressed send_response tool={tool_call.name} "
                f"id={tool_call.id} reason=cancelled_after_completion"
            )
            self._cancelled_ids.pop(tool_call.id, None)
            self._finish_hash(tool_hash)
            return

        await self._send_tool_result(ToolHandlerResult(
            action=ToolResponseAction.SEND_RESPONSE,
            tool_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
        ))
        self._finish_hash(tool_hash)
        await self.on_complete(tool_call, result)

    async def _handle_non_blocking(
        self, tool_call: ToolCallData, config: ToolConfig
    ) -> None:
        logger.info(f"[ToolHandler] Scheduling tool={tool_call.name} id={tool_call.id} mode=non-blocking")

        await self._send_tool_result(ToolHandlerResult(
            action=ToolResponseAction.SEND_INTERIM,
            tool_id=tool_call.id,
            tool_name=tool_call.name,
            interim_message=config.interim_message,
        ))

        task = asyncio.create_task(
            self._run_non_blocking(tool_call, config.execution_delay)
        )
        self._pending_tasks[tool_call.id] = task

    async def _run_non_blocking(
        self, tool_call: ToolCallData, delay: float
    ) -> None:
        tool_hash = self._compute_hash(tool_call.name, tool_call.args)
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            result = await self._execute(tool_call)
        except asyncio.CancelledError:
            logger.info(f"[ToolHandler] Cancelled tool={tool_call.name} id={tool_call.id}")
            self._finish_hash(tool_hash)
            return
        except Exception as e:
            logger.error(
                "[ToolHandler] Execution failed tool=%s mode=non-blocking: %s",
                tool_call.name, e, exc_info=True,
            )
            result = {"success": False, "error": str(e)}

        if tool_call.id in self._cancelled_ids:
            logger.info(
                f"[ToolHandler] Suppressed send_context tool={tool_call.name} "
                f"id={tool_call.id} reason=cancelled_after_completion"
            )
            self._cancelled_ids.pop(tool_call.id, None)
            self._finish_hash(tool_hash)
            self._pending_tasks.pop(tool_call.id, None)
            return

        await self._send_tool_result(ToolHandlerResult(
            action=ToolResponseAction.SEND_CONTEXT,
            tool_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
        ))
        self._finish_hash(tool_hash)
        self._pending_tasks.pop(tool_call.id, None)
        await self.on_complete(tool_call, result)

    async def _execute(self, tool_call: ToolCallData) -> Dict[str, Any]:
        """Invoke the decorated tool method."""
        method = getattr(self, tool_call.name)
        return await method(**tool_call.args)

    def _is_duplicate(self, tool_call: ToolCallData) -> bool:
        now = time.monotonic()
        self._purge_expired(now)

        if tool_call.id in self._processed_tool_ids:
            logger.warning(f"[ToolHandler] Duplicate skipped tool={tool_call.name} id={tool_call.id} reason=same_id")
            return True

        tool_hash = self._compute_hash(tool_call.name, tool_call.args)
        if tool_hash in self._in_flight_hashes and now < self._in_flight_hashes[tool_hash]:
            logger.warning(f"[ToolHandler] Duplicate skipped tool={tool_call.name} id={tool_call.id} reason=same_content")
            return True

        self._processed_tool_ids[tool_call.id] = now
        self._in_flight_hashes[tool_hash] = float("inf")
        return False

    def _finish_hash(self, tool_hash: str) -> None:
        if tool_hash in self._in_flight_hashes:
            self._in_flight_hashes[tool_hash] = (
                time.monotonic() + DEDUP_COOLDOWN_SECONDS
            )

    @staticmethod
    def _compute_hash(tool_name: str, args: dict) -> str:
        args_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(f"{tool_name}:{args_str}".encode()).hexdigest()[:16]

    def _purge_expired(self, now: float) -> None:
        """Expire dedup bookkeeping so long-lived sessions cannot leak."""
        expired_hashes = [h for h, exp in self._in_flight_hashes.items() if now >= exp]
        for h in expired_hashes:
            del self._in_flight_hashes[h]

        cutoff = now - DEDUP_COOLDOWN_SECONDS
        expired_ids = [tid for tid, ts in self._processed_tool_ids.items() if ts < cutoff]
        for tid in expired_ids:
            del self._processed_tool_ids[tid]

        expired_cancellations = [tid for tid, ts in self._cancelled_ids.items() if ts < cutoff]
        for tid in expired_cancellations:
            del self._cancelled_ids[tid]
