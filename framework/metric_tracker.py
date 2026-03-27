"""MetricTracker — built-in session instrumentation for the orchestrator."""

import logging
import time
from typing import Optional

from .models import UsageMetadataData

logger = logging.getLogger(__name__)


class MetricTracker:
    def __init__(self):
        self.audio_packets_sent: int = 0
        self.audio_packets_received: int = 0
        self.user_turns: int = 0
        self.model_turns: int = 0
        self.interruptions: int = 0
        self.tool_calls: int = 0
        self.user_word_count: int = 0
        self.model_word_count: int = 0
        self.total_usage: UsageMetadataData = UsageMetadataData()
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None

    @property
    def total_duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return end - self.started_at

    def start(self) -> None:
        self.started_at = time.monotonic()

    def stop(self) -> None:
        self.ended_at = time.monotonic()

    def on_audio_sent(self) -> None:
        self.audio_packets_sent += 1

    def on_audio_received(self) -> None:
        self.audio_packets_received += 1

    def on_user_turn(self) -> None:
        self.user_turns += 1

    def on_model_turn(self) -> None:
        self.model_turns += 1

    def on_interruption(self) -> None:
        self.interruptions += 1

    def on_tool_call(self) -> None:
        self.tool_calls += 1

    def on_user_transcript(self, text: str) -> None:
        self.user_word_count += len(text.split())

    def on_model_transcript(self, text: str) -> None:
        self.model_word_count += len(text.split())

    def on_usage_metadata(self, data: UsageMetadataData) -> None:
        self.total_usage.prompt_token_count += data.prompt_token_count
        self.total_usage.response_token_count += data.response_token_count
        self.total_usage.total_token_count += data.total_token_count
        self.total_usage.thoughts_token_count += data.thoughts_token_count
        self.total_usage.tool_use_prompt_token_count += data.tool_use_prompt_token_count

    def to_dict(self) -> dict:
        return {
            "audio_packets_sent": self.audio_packets_sent,
            "audio_packets_received": self.audio_packets_received,
            "user_turns": self.user_turns,
            "model_turns": self.model_turns,
            "interruptions": self.interruptions,
            "tool_calls": self.tool_calls,
            "user_word_count": self.user_word_count,
            "model_word_count": self.model_word_count,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "total_usage": self.total_usage.model_dump(),
        }
