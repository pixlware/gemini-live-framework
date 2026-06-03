"""Tracks which party owns the conversation turn and drives the idle timers.

Single source of truth for conversation state. Owns the two idle timers
and derives their operations from a state machine that validates END
events against the current state. This structurally prevents cross-source
VAD reorder (user VAD from Gemini vs model VAD from the local detector)
from corrupting timer state — a stale ``user_speech_end`` arriving after
a ``model_speech_start`` is dropped instead of arming the model-idle
timer mid-utterance.
"""

from __future__ import annotations

from enum import Enum
from typing import Awaitable, Callable, Optional

from .timer import Timer
from .logger import SessionLogger


class ConversationState(Enum):
    INITIAL = "initial"
    USER_TALKING = "user_talking"
    MODEL_TALKING = "model_talking"
    WAITING_FOR_MODEL = "waiting_for_model"
    WAITING_FOR_USER = "waiting_for_user"


OnStateChange = Callable[[ConversationState, ConversationState], Awaitable[None]]


class TurnTracker:
    """State machine for conversation turns.

    Methods to call from VAD handlers:

        user_speech_start() / user_speech_end()
        model_speech_start() / model_speech_end()

    END events are silently dropped unless the matching source currently
    owns the state. START events are always accepted (supports barge-in).
    """

    def __init__(
        self,
        user_idle_timer: Optional[Timer] = None,
        model_idle_timer: Optional[Timer] = None,
        on_state_change: Optional[OnStateChange] = None,
        logger: Optional[SessionLogger] = None,
    ) -> None:
        self._user_idle_timer = user_idle_timer
        self._model_idle_timer = model_idle_timer
        self._on_state_change = on_state_change
        self._state = ConversationState.INITIAL
        self._session_logger = logger

        if logger:
            if self._user_idle_timer:
                self._user_idle_timer.logger = logger
            if self._model_idle_timer:
                self._model_idle_timer.logger = logger

    @property
    def logger(self) -> SessionLogger:
        if self._session_logger is None:
            self._session_logger = SessionLogger()
        return self._session_logger

    @logger.setter
    def logger(self, logger: SessionLogger) -> None:
        self._session_logger = logger
        if self._user_idle_timer:
            self._user_idle_timer.logger = logger
        if self._model_idle_timer:
            self._model_idle_timer.logger = logger

    @property
    def user_idle_timer(self) -> Optional[Timer]:
        return self._user_idle_timer

    @user_idle_timer.setter
    def user_idle_timer(self, timer: Optional[Timer]) -> None:
        self._user_idle_timer = timer
        if timer and self._session_logger:
            timer.logger = self._session_logger

    @property
    def model_idle_timer(self) -> Optional[Timer]:
        return self._model_idle_timer

    @model_idle_timer.setter
    def model_idle_timer(self, timer: Optional[Timer]) -> None:
        self._model_idle_timer = timer
        if timer and self._session_logger:
            timer.logger = self._session_logger

    @property
    def state(self) -> ConversationState:
        return self._state

    async def start(self) -> None:
        self._state = ConversationState.INITIAL
        if self._user_idle_timer:
            self._user_idle_timer.reset()
        if self._model_idle_timer:
            self._model_idle_timer.reset()

    async def stop(self) -> None:
        if self._user_idle_timer:
            self._user_idle_timer.reset()
        if self._model_idle_timer:
            self._model_idle_timer.reset()

    async def user_speech_start(self) -> None:
        if self._state == ConversationState.USER_TALKING:
            return
        if self._user_idle_timer:
            self._user_idle_timer.reset()
        if self._model_idle_timer:
            self._model_idle_timer.reset()
        await self._transition(ConversationState.USER_TALKING)

    async def user_speech_end(self) -> None:
        if self._state != ConversationState.USER_TALKING:
            return
        if self._model_idle_timer:
            self._model_idle_timer.start()
        await self._transition(ConversationState.WAITING_FOR_MODEL)

    async def model_speech_start(self) -> None:
        if self._state == ConversationState.MODEL_TALKING:
            return
        if self._model_idle_timer:
            self._model_idle_timer.reset()
        if self._user_idle_timer:
            self._user_idle_timer.stop()
        await self._transition(ConversationState.MODEL_TALKING)

    async def model_speech_end(self) -> None:
        if self._state != ConversationState.MODEL_TALKING:
            return
        if self._user_idle_timer:
            self._user_idle_timer.start()
        await self._transition(ConversationState.WAITING_FOR_USER)

    async def bot_timeout(self) -> None:
        """Transition state to WAITING_FOR_USER and start the user idle timer after a bot idle timeout."""
        if self._state != ConversationState.WAITING_FOR_MODEL:
            return
        if self._model_idle_timer:
            self._model_idle_timer.reset()
        if self._user_idle_timer:
            self._user_idle_timer.start()
        await self._transition(ConversationState.WAITING_FOR_USER)

    async def _transition(self, new_state: ConversationState) -> None:
        old_state = self._state
        if old_state == new_state:
            return
        self._state = new_state
        self.logger.debug(
            f"[TurnTracker] Turn state changed: {old_state.value} -> {new_state.value}",
            old_turn_state=old_state.value,
            new_turn_state=new_state.value,
        )
        if self._on_state_change:
            await self._on_state_change(old_state, new_state)
