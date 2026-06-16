"""Local Voice Activity Detection using Silero VAD via sherpa-onnx."""

from __future__ import annotations

import os
from typing import Generator, Optional, Tuple

import numpy as np
import sherpa_onnx


class SileroVad:
    """Generator-based Silero VAD using sherpa-onnx."""

    def __init__(
        self,
        threshold: float = 0.5,
        prefix_padding_ms: int = 320,
        silence_duration_ms: int = 600,
        min_speech_duration_ms: int = 200,
        max_speech_duration_ms: int = 60000,
    ):
        model_path = os.path.join(os.path.dirname(__file__), "models", "silero_vad.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[SileroVad] Model not found: {model_path}")

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = model_path
        config.silero_vad.threshold = threshold
        config.silero_vad.min_silence_duration = silence_duration_ms / 1000.0
        config.silero_vad.min_speech_duration = min_speech_duration_ms / 1000.0
        config.silero_vad.max_speech_duration = max_speech_duration_ms / 1000.0
        config.sample_rate = 16000

        self._vad = sherpa_onnx.VoiceActivityDetector(config)
        self._frame_bytes = config.silero_vad.window_size * 2  # PCM16

        self._max_prefix_bytes = int(16000 * 2 * prefix_padding_ms / 1000)
        self._pre_speech_buf = bytearray()
        self._accumulator = bytearray()
        self._is_speaking = False

    def process_audio(
        self, chunk: bytes
    ) -> Generator[Tuple[Optional[str], bytes], None, None]:
        """Process arbitrary PCM16 16 kHz audio and yield VAD signals with gated audio.

        Yields:
            Tuple[Optional[str], bytes]:
                - "start" when speech begins (audio includes pre-speech buffer).
                - "end" when speech ends.
                - None for continuous speech frames.
        """
        self._accumulator.extend(chunk)

        while len(self._accumulator) >= self._frame_bytes:
            frame = bytes(self._accumulator[: self._frame_bytes])
            del self._accumulator[: self._frame_bytes]

            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            self._vad.accept_waveform(samples)

            while not self._vad.empty():
                self._vad.pop()

            is_speech = self._vad.is_speech_detected()

            if is_speech and not self._is_speaking:
                self._is_speaking = True
                audio = bytes(self._pre_speech_buf) + frame
                self._pre_speech_buf.clear()
                yield ("start", audio)

            elif is_speech:
                yield (None, frame)

            elif not is_speech and self._is_speaking:
                self._is_speaking = False
                yield ("end", b"")

            else:
                self._pre_speech_buf.extend(frame)
                if len(self._pre_speech_buf) > self._max_prefix_bytes:
                    del self._pre_speech_buf[: -self._max_prefix_bytes]
