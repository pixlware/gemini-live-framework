"""Audio recorder — captures user and model audio into a wall-clock aligned stereo file."""

from __future__ import annotations

import datetime
import os
import time
import uuid
import io
import wave
from typing import Optional

import numpy as np
from google.cloud import storage

from config import settings
from .audio_transcoder import PcmResampler
from .logger import SessionLogger

BYTES_PER_SAMPLE = 2  # 16-bit PCM
USER_SAMPLE_RATE = 16_000
MODEL_SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = USER_SAMPLE_RATE
NUM_CHANNELS_STEREO = 2


class AudioRecorder:
    """Records user and model audio into a wall-clock aligned stereo WAV file."""

    def __init__(
        self,
        filename: str = "",
        output_dir: str = "",
        storage_type: str = "local",  # "local" or "cloud"
        bucket_name: Optional[str] = None,
        logger: Optional[SessionLogger] = None,
    ):
        self.filename = filename or uuid.uuid4().hex
        self._output_dir = output_dir or ".recordings"

        self._start_mono: float = 0.0
        self._start_time: datetime.datetime | None = None

        self._user_track = bytearray()
        self._model_track = bytearray()

        self._model_resampler = PcmResampler(MODEL_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)

        self.is_recording = False
        self._session_logger = logger

        self._storage_type = storage_type
        self._bucket_name = bucket_name or settings.GCS_BUCKET_NAME

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
        self._append_to_track(self._user_track, audio_data)

    def record_model_audio(self, audio_data: bytes) -> None:
        """Append a chunk of model audio (PCM16 @ 24 kHz)."""
        if not self.is_recording:
            return
        self._append_to_track(self._model_track, self._model_resampler.process(audio_data))

    def stop(self) -> None:
        """Finalize the recording: mix stereo, build WAV, and persist."""
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

            stereo = self._mix_stereo(
                bytes(self._user_track), bytes(self._model_track)
            )

            duration_sec = max_len / (OUTPUT_SAMPLE_RATE * BYTES_PER_SAMPLE)
            wav_bytes = self._build_wav_bytes(stereo)
            filepath = self._save_recording(wav_bytes)

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

    def _save_recording(self, wav_bytes: bytes) -> str:
        """Route to local or cloud storage based on ``storage_type``."""
        if self._storage_type == "cloud":
            return self._save_cloud(wav_bytes)
        return self._save_local(wav_bytes)

    def _save_local(self, wav_bytes: bytes) -> str:
        """Write the WAV bytes to local disk. Returns the file path."""
        os.makedirs(self._output_dir, exist_ok=True)
        filepath = os.path.join(self._output_dir, f"{self.filename}.wav")
        with open(filepath, "wb") as f:
            f.write(wav_bytes)
        return filepath

    def _save_cloud(self, wav_bytes: bytes) -> str:
        """Upload the WAV bytes to Google Cloud Storage. Returns the GCS URI.

        Subclasses override this method to upload to a different cloud
        provider (e.g. S3).
        """
        bucket_name = self._bucket_name
        if not bucket_name:
            raise ValueError("GCS bucket name is not configured.")

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        gcs_prefix = self._output_dir.strip("./")
        gcs_path = f"{gcs_prefix}/{self.filename}.wav" if gcs_prefix else f"{self.filename}.wav"

        blob = bucket.blob(gcs_path)
        blob.upload_from_string(wav_bytes, content_type="audio/wav")

        return f"gs://{bucket_name}/{gcs_path}"

    def _build_wav_bytes(self, audio_data: bytes) -> bytes:
        """Create a full WAV file in memory from raw stereo PCM data."""
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(NUM_CHANNELS_STEREO)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(OUTPUT_SAMPLE_RATE)
            wf.writeframes(audio_data)
        return wav_io.getvalue()

    def _append_to_track(self, track: bytearray, audio: bytes) -> None:
        elapsed = time.monotonic() - self._start_mono
        expected_bytes = int(elapsed * OUTPUT_SAMPLE_RATE * BYTES_PER_SAMPLE)
        expected_bytes -= expected_bytes % BYTES_PER_SAMPLE

        if len(track) < expected_bytes:
            track.extend(b"\x00" * (expected_bytes - len(track)))

        track.extend(audio)

    @staticmethod
    def _mix_stereo(track_user: bytes, track_model: bytes) -> bytes:
        """Interleave two mono PCM16 byte strings into a single stereo PCM16 stream.

        Left Channel = User
        Right Channel = Model
        """
        user_samples = np.frombuffer(track_user, dtype=np.int16)
        model_samples = np.frombuffer(track_model, dtype=np.int16)
        stereo_samples = np.column_stack((user_samples, model_samples)).flatten()
        return stereo_samples.tobytes()
