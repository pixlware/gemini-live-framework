"""Modular audio transcoders for format/rate conversion in the transport layer."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

try:
    import audioop
except ImportError:
    import audioop_lts as audioop  # type: ignore[no-redef]

from .models import AudioFormat

logger = logging.getLogger(__name__)

PCM16_SAMPLE_WIDTH = 2
MONO_CHANNELS = 1
MULAW_SAMPLE_RATE = 8000


class AudioTranscoder(ABC):
    """Common interface for all audio transcoders."""

    @abstractmethod
    def process(self, chunk: bytes) -> bytes:
        """Transform a chunk of audio bytes. May return fewer or more bytes than input."""
        ...

    def reset(self) -> None:
        """Reset any internal state (e.g. between conversations)."""


class PcmResampler(AudioTranscoder):
    """Stateful PCM16 mono sample-rate converter."""

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self._src_rate = src_rate
        self._dst_rate = dst_rate
        self._state: object | None = None
        self._noop = src_rate == dst_rate

    def process(self, chunk: bytes) -> bytes:
        if self._noop or not chunk:
            return chunk
        out, self._state = audioop.ratecv(
            chunk, PCM16_SAMPLE_WIDTH, MONO_CHANNELS,
            self._src_rate, self._dst_rate, self._state,
        )
        return out

    def reset(self) -> None:
        self._state = None


class MulawDecoder(AudioTranscoder):
    """Mulaw 8kHz -> PCM16 at a target sample rate."""

    def __init__(self, target_pcm_rate: int) -> None:
        self._resampler = PcmResampler(MULAW_SAMPLE_RATE, target_pcm_rate)

    def process(self, chunk: bytes) -> bytes:
        if not chunk:
            return chunk
        pcm_8k = audioop.ulaw2lin(chunk, PCM16_SAMPLE_WIDTH)
        return self._resampler.process(pcm_8k)

    def reset(self) -> None:
        self._resampler.reset()


class MulawEncoder(AudioTranscoder):
    """PCM16 at a source sample rate -> mulaw 8kHz."""

    def __init__(self, source_pcm_rate: int) -> None:
        self._resampler = PcmResampler(source_pcm_rate, MULAW_SAMPLE_RATE)

    def process(self, chunk: bytes) -> bytes:
        if not chunk:
            return chunk
        pcm_8k = self._resampler.process(chunk)
        return audioop.lin2ulaw(pcm_8k, PCM16_SAMPLE_WIDTH)

    def reset(self) -> None:
        self._resampler.reset()


def build_transcoder(
    src_format: AudioFormat,
    src_rate: int,
    dst_format: AudioFormat,
    dst_rate: int,
) -> Optional[AudioTranscoder]:
    """Return the appropriate transcoder, or None when no conversion is needed."""
    if src_format == dst_format == AudioFormat.PCM16:
        if src_rate == dst_rate:
            return None
        return PcmResampler(src_rate, dst_rate)

    if src_format == AudioFormat.MULAW and dst_format == AudioFormat.PCM16:
        return MulawDecoder(dst_rate)

    if src_format == AudioFormat.PCM16 and dst_format == AudioFormat.MULAW:
        return MulawEncoder(src_rate)

    if src_format == dst_format == AudioFormat.MULAW:
        return None

    raise ValueError(
        f"[AudioTranscoder] Unsupported transcoding: {src_format.value}@{src_rate} -> {dst_format.value}@{dst_rate}"
    )
