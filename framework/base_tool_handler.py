"""
Tool handler base class for the Gemini Live Framework.

Users subclass BaseToolHandler and define async methods whose names match
their Gemini function declarations.  The framework dispatches tool calls
to these methods automatically via getattr.

The handler is fully self-contained: it owns dedup, blocking/non-blocking
scheduling, and background task management.  It communicates results back
to the Orchestrator through an async result queue — the Orchestrator never
touches tool execution internals, it just reads structured results and
forwards them to Gemini.

Example
-------
    from framework.base_tool_handler import BaseToolHandler, tool

    class MyTools(BaseToolHandler):

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
from typing import Any, Callable, Dict, List, Optional

from .models import ToolCallData

logger = logging.getLogger(__name__)

DEDUP_COOLDOWN_SECONDS = 15.0


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
    """Decorator that attaches a ToolConfig to a handler method.

    Methods without the decorator default to blocking execution.
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
    """Structured result pushed to the queue for the Orchestrator to consume."""
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
    blocking / non-blocking dispatch → result queuing.  The Orchestrator
    reads ``result_queue`` and translates each ``ToolHandlerResult`` into
    the appropriate Gemini API call.
    """

    def __init__(self):
        self.result_queue: asyncio.Queue[Optional[ToolHandlerResult]] = asyncio.Queue()
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        self._processed_tool_ids: Dict[str, float] = {}
        self._in_flight_hashes: Dict[str, float] = {}

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
        """Process an incoming tool call (dedup → dispatch → queue result)."""
        if self._is_duplicate(tool_call):
            return

        config = self._get_tool_config(tool_call.name)

        if config.blocking:
            await self._handle_blocking(tool_call)
        else:
            await self._handle_non_blocking(tool_call, config)

    async def handle_cancellation(self, tool_ids: List[str]) -> None:
        """Cancel pending (non-blocking) tool tasks and notify subclass."""
        cancelled = 0
        for tool_id in tool_ids:
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
        await self.result_queue.put(None)

    # --- Internals -------------------------------------------------------

    async def _execute(self, tool_call: ToolCallData) -> Dict[str, Any]:
        """Look up ``self.<tool_call.name>`` and call it with ``**tool_call.args``."""
        method = getattr(self, tool_call.name, None)
        if method is None or not callable(method):
            logger.warning(f"[ToolHandler] No method found tool={tool_call.name}")
            return {
                "success": False,
                "error": f"Unknown tool: {tool_call.name}",
            }
        return await method(**tool_call.args)

    def _get_tool_config(self, tool_name: str) -> ToolConfig:
        """Read ``@tool(...)`` decorator metadata, fall back to default (blocking) config."""
        method = getattr(self, tool_name, None)
        if method and hasattr(method, "_tool_config"):
            return method._tool_config
        return ToolConfig()

    async def _handle_blocking(self, tool_call: ToolCallData) -> None:
        tool_hash = self._compute_hash(tool_call.name, tool_call.args)
        logger.info(f"[ToolHandler] Executing tool={tool_call.name} id={tool_call.id} mode=blocking")
        try:
            result = await self._execute(tool_call)
        except Exception as e:
            logger.error(f"[ToolHandler] Execution failed tool={tool_call.name} mode=blocking: {e}")
            result = {"success": False, "error": str(e)}

        await self.result_queue.put(ToolHandlerResult(
            action=ToolResponseAction.SEND_RESPONSE,
            tool_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
        ))
        self._finish_hash(tool_hash, success=result.get("success") is not False)
        logger.info(f"[ToolHandler] Result queued tool={tool_call.name} action=send_response")
        await self.on_complete(tool_call, result)

    async def _handle_non_blocking(
        self, tool_call: ToolCallData, config: ToolConfig
    ) -> None:
        logger.info(f"[ToolHandler] Scheduling tool={tool_call.name} id={tool_call.id} mode=non-blocking")

        await self.result_queue.put(ToolHandlerResult(
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
            self._finish_hash(tool_hash, success=False)
            return
        except Exception as e:
            logger.error(f"[ToolHandler] Execution failed tool={tool_call.name} mode=non-blocking: {e}")
            result = {"success": False, "error": str(e)}

        await self.result_queue.put(ToolHandlerResult(
            action=ToolResponseAction.SEND_CONTEXT,
            tool_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
        ))
        self._finish_hash(tool_hash, success=result.get("success") is not False)
        self._pending_tasks.pop(tool_call.id, None)
        logger.info(f"[ToolHandler] Result queued tool={tool_call.name} action=send_context")
        await self.on_complete(tool_call, result)

    def _is_duplicate(self, tool_call: ToolCallData) -> bool:
        now = time.monotonic()
        self._purge_expired_hashes(now)

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

    def _finish_hash(self, tool_hash: str, success: bool) -> None:
        if tool_hash in self._in_flight_hashes:
            cooldown = DEDUP_COOLDOWN_SECONDS if success else 1.0
            self._in_flight_hashes[tool_hash] = time.monotonic() + cooldown

    @staticmethod
    def _compute_hash(tool_name: str, args: dict) -> str:
        args_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(f"{tool_name}:{args_str}".encode()).hexdigest()[:16]

    def _purge_expired_hashes(self, now: float) -> None:
        expired = [h for h, exp in self._in_flight_hashes.items() if now >= exp]
        for h in expired:
            del self._in_flight_hashes[h]
