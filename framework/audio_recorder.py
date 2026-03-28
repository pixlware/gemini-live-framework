"""Audio recorder — captures user and model audio into a wall-clock aligned mono file.

Both tracks are resampled to 24 kHz PCM16, silence-filled for temporal
alignment, and mixed down into a single mono channel.
"""

from __future__ import annotations

import array
import datetime
import io
import logging
import os
import time
import uuid
import wave
from enum import Enum
from typing import Optional

from .audio_transcoder import PcmResampler

logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2  # 16-bit PCM
OUTPUT_SAMPLE_RATE = 24_000
NUM_CHANNELS_MONO = 1


class RecordingFormat(Enum):
    WAV = "wav"
    MP3 = "mp3"


class AudioRecorder:
    """Records user and model audio into a mono file.

    Both tracks are wall-clock aligned at 24 kHz PCM16, then mixed
    down into a single channel.  Gaps in either track are filled with
    silence so that temporal alignment is preserved.
    """

    def __init__(
        self,
        output_dir: str = ".recordings",
        output_format: RecordingFormat = RecordingFormat.WAV,
    ):
        self._output_dir = output_dir
        self._output_format = output_format
        self.filename: str = uuid.uuid4().hex

        self._start_mono: Optional[float] = None
        self._start_wall: Optional[datetime.datetime] = None
        self._end_wall: Optional[datetime.datetime] = None

        self._user_track = bytearray()
        self._model_track = bytearray()

        self._user_resampler: Optional[PcmResampler] = None
        self._model_resampler: Optional[PcmResampler] = None

        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start(self) -> None:
        """Begin recording.  Call once when the call starts."""
        self._start_mono = time.monotonic()
        self._start_wall = datetime.datetime.now(datetime.timezone.utc)
        self._user_track = bytearray()
        self._model_track = bytearray()
        self._is_recording = True

        logger.info("[AudioRecorder] Recording started")

    def record_user_audio(self, audio_data: bytes, sample_rate: int) -> None:
        """Append a chunk of user audio."""
        if not self._is_recording:
            return
        resampled = self._ensure_resampled_user(audio_data, sample_rate)
        self._append_to_track(self._user_track, resampled)

    def record_model_audio(self, audio_data: bytes, sample_rate: int) -> None:
        """Append a chunk of model audio."""
        if not self._is_recording:
            return
        resampled = self._ensure_resampled_model(audio_data, sample_rate)
        self._append_to_track(self._model_track, resampled)

    def stop(self) -> Optional[str]:
        """Finalize the recording and write to disk.

        Returns the file path on success, or ``None`` if nothing was recorded
        or an error occurred.
        """
        if not self._is_recording:
            return None

        self._is_recording = False
        self._end_wall = datetime.datetime.now(datetime.timezone.utc)

        if not self._user_track and not self._model_track:
            logger.warning("[AudioRecorder] No audio captured — skipping write")
            return None

        try:
            max_len = max(len(self._user_track), len(self._model_track))
            max_len += max_len % BYTES_PER_SAMPLE  # align to sample boundary
            self._user_track.extend(b"\x00" * (max_len - len(self._user_track)))
            self._model_track.extend(b"\x00" * (max_len - len(self._model_track)))

            mono = self._mix_mono(
                bytes(self._user_track), bytes(self._model_track)
            )

            duration_sec = max_len / (OUTPUT_SAMPLE_RATE * BYTES_PER_SAMPLE)
            filepath = self._save_recording(mono)

            logger.info(
                "[AudioRecorder] Recording saved: path=%s, start=%s, end=%s, duration=%.1fs",
                filepath,
                self._start_wall.isoformat() if self._start_wall else "?",
                self._end_wall.isoformat() if self._end_wall else "?",
                duration_sec,
            )

            return filepath
        except Exception as exc:
            logger.error("[AudioRecorder] Failed to save recording: %s", exc, exc_info=True)
            return None
        finally:
            self._user_track = bytearray()
            self._model_track = bytearray()

    # --- Storage -------------------------------------------------------

    def _save_recording(self, audio_data: bytes) -> str:
        """Persist the mono audio to local disk.  Returns the file path."""
        os.makedirs(self._output_dir, exist_ok=True)

        ext = self._output_format.value
        filepath = os.path.join(self._output_dir, f"{self.filename}.{ext}")

        if self._output_format == RecordingFormat.WAV:
            self._write_wav(filepath, audio_data)
        else:
            self._write_mp3(filepath, audio_data)

        return filepath

    # --- Internal helpers -----------------------------------------------

    def _ensure_resampled_user(self, data: bytes, sample_rate: int) -> bytes:
        if sample_rate == OUTPUT_SAMPLE_RATE:
            return data
        if self._user_resampler is None:
            self._user_resampler = PcmResampler(sample_rate, OUTPUT_SAMPLE_RATE)
        return self._user_resampler.process(data)

    def _ensure_resampled_model(self, data: bytes, sample_rate: int) -> bytes:
        if sample_rate == OUTPUT_SAMPLE_RATE:
            return data
        if self._model_resampler is None:
            self._model_resampler = PcmResampler(sample_rate, OUTPUT_SAMPLE_RATE)
        return self._model_resampler.process(data)

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
        a = array.array("h")
        a.frombytes(track_a)
        b = array.array("h")
        b.frombytes(track_b)

        mixed = array.array("h", (
            max(-32768, min(32767, a[i] + b[i]))
            for i in range(len(a))
        ))
        return mixed.tobytes()

    @staticmethod
    def _write_wav(filepath: str, audio_data: bytes) -> None:
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(NUM_CHANNELS_MONO)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(OUTPUT_SAMPLE_RATE)
            wf.writeframes(audio_data)

    @staticmethod
    def _write_mp3(filepath: str, audio_data: bytes) -> None:
        try:
            from pydub import AudioSegment  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "MP3 output requires 'pydub' (and ffmpeg). "
                "Install with:  pip install pydub"
            ) from None

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(NUM_CHANNELS_MONO)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(OUTPUT_SAMPLE_RATE)
            wf.writeframes(audio_data)
        wav_buf.seek(0)

        audio_seg = AudioSegment.from_wav(wav_buf)
        audio_seg.export(filepath, format="mp3")
