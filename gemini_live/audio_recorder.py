"""Audio recorder — captures user and model audio into a wall-clock aligned mono file."""

from __future__ import annotations

import datetime
import os
import time
import uuid
import wave
from typing import Optional

import numpy as np

from .audio_transcoder import PcmResampler
from .logger import SessionLogger

BYTES_PER_SAMPLE = 2  # 16-bit PCM
USER_SAMPLE_RATE = 16_000
MODEL_SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = MODEL_SAMPLE_RATE
NUM_CHANNELS_MONO = 1


class AudioRecorder:
    """Records user and model audio into a wall-clock aligned mono WAV file."""

    def __init__(
        self,
        output_dir: str = ".recordings",
        logger: Optional[SessionLogger] = None,
    ):
        self._output_dir = output_dir
        self.filename: str = uuid.uuid4().hex

        self._start_mono: float = 0.0
        self._start_time: datetime.datetime | None = None

        self._user_track = bytearray()
        self._model_track = bytearray()

        self._user_resampler = PcmResampler(USER_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

        self.is_recording = False
        self._session_logger = logger

    @property
    def logger(self) -> SessionLogger:
        if self._session_logger is None:
            self._session_logger = SessionLogger()
        return self._session_logger

    @logger.setter
    def logger(self, logger: SessionLogger) -> None:
        self._session_logger = logger

    def start(self) -> None:
        """Begin recording. Call once per session."""
        self._start_mono = time.monotonic()
        self._start_time = datetime.datetime.now(datetime.timezone.utc)
        self.is_recording = True

        self.logger.info("[AudioRecorder] Recording started")

    def record_user_audio(self, audio_data: bytes) -> None:
        """Append a chunk of user audio (PCM16 @ 16 kHz)."""
        if not self.is_recording:
            return
        self._append_to_track(self._user_track, self._user_resampler.process(audio_data))

    def record_model_audio(self, audio_data: bytes) -> None:
        """Append a chunk of model audio (PCM16 @ 24 kHz)."""
        if not self.is_recording:
            return
        self._append_to_track(self._model_track, audio_data)

    def stop(self) -> None:
        """Finalize the recording and write to disk."""
        if not self.is_recording:
            return

        self.is_recording = False

        if not self._user_track and not self._model_track:
            self.logger.warning("[AudioRecorder] No audio captured — skipping write")
            return

        try:
            max_len = max(len(self._user_track), len(self._model_track))
            self._user_track.extend(b"\x00" * (max_len - len(self._user_track)))
            self._model_track.extend(b"\x00" * (max_len - len(self._model_track)))

            mono = self._mix_mono(
                bytes(self._user_track), bytes(self._model_track)
            )

            duration_sec = max_len / (OUTPUT_SAMPLE_RATE * BYTES_PER_SAMPLE)
            filepath = self._save_recording(mono)
            start_time = self._start_time.isoformat() if self._start_time else "?"
            end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            info = (
                f"path={filepath}",
                f"start={start_time}",
                f"end={end_time}",
                f"duration={duration_sec:.1f}s",
            )

            self.logger.info(f"[AudioRecorder] Recording saved: {info}")
        except Exception as exc:
            self.logger.error(
                "[AudioRecorder] Failed to save recording; audio for this call will be DISCARDED: %s",
                error=str(exc),
            )
        finally:
            self._user_track = bytearray()
            self._model_track = bytearray()

    # --- Storage -------------------------------------------------------

    def _save_recording(self, audio_data: bytes) -> str:
        """Persist the mono audio to local disk. Returns the file path."""
        os.makedirs(self._output_dir, exist_ok=True)
        filepath = os.path.join(self._output_dir, f"{self.filename}.wav")
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(NUM_CHANNELS_MONO)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(OUTPUT_SAMPLE_RATE)
            wf.writeframes(audio_data)
        return filepath

    def _append_to_track(self, track: bytearray, audio: bytes) -> None:
        elapsed = time.monotonic() - self._start_mono
        expected_bytes = int(elapsed * OUTPUT_SAMPLE_RATE * BYTES_PER_SAMPLE)
        expected_bytes -= expected_bytes % BYTES_PER_SAMPLE

        if len(track) < expected_bytes:
            track.extend(b"\x00" * (expected_bytes - len(track)))

        track.extend(audio)

    @staticmethod
    def _mix_mono(track_a: bytes, track_b: bytes) -> bytes:
        """Mix two equal-length mono PCM16 byte strings into one mono stream."""
        a = np.frombuffer(track_a, dtype=np.int16).astype(np.int32)
        b = np.frombuffer(track_b, dtype=np.int16).astype(np.int32)
        mixed = np.clip(a + b, -32768, 32767).astype(np.int16)
        return mixed.tobytes()
