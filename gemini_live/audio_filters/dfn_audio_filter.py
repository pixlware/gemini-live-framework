"""
ONNX-based DeepFilterNet streaming audio filter for uplink noise suppression.

Requires ``dfnstream-py==0.2.1``; operates in BYPASS mode if missing.

Pipeline: 16 kHz PCM16 → float32 → upsample 48 kHz → DFN denoise → downsample
16 kHz → clip & quantize int16 → output.

All resampling is done in float32 to avoid quantisation noise from int16 round-trips.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base_audio_filter import BaseAudioFilter

logger = logging.getLogger(__name__)

try:
    from dfnstream_py import DeepFilterNetStreamingONNX
    import soxr
    _HAS_DFN = True
except ImportError:
    _HAS_DFN = False


class DeepFilterNetAudioFilter(BaseAudioFilter):
    """
    Real-time ONNX-based DeepFilterNet denoiser for 16-bit PCM mono at 16 kHz.
    Uses soxr.ResampleStream for stateful 16kHz<->48kHz resampling with phase
    continuity between blocks and deterministic output sizes.
    """

    def __init__(
        self,
        atten_lim: float = 15.0,
        resampling_quality: str = "HQ",
    ) -> None:
        """Create the filter and streaming resamplers.

        Args:
            atten_lim: DeepFilterNet attenuation limit in dB. Caps how strongly
                the model can suppress non-speech content. ``10`` is a solid
                default for **telephony**; ``15`` (default) is often better for
                **web or app** microphones where you want stronger denoising.
            resampling_quality: soxr preset for 16 kHz ↔ 48 kHz, one of
                ``"QQ"``, ``"LQ"``, ``"MQ"``, ``"HQ"``, ``"VHQ"``. ``"HQ"``
                (default) balances quality and CPU for real-time **telephony**;
                ``"VHQ"`` trades more CPU for finer resampling, which suits many
                **web or app** uplinks.
        """
        if not _HAS_DFN:
            self.enabled = False
            logger.warning(
                "[DeepFilterNetAudioFilter] Missing 'dfnstream-py' dependency. "
                "Filter will operate in BYPASS mode."
            )
            return

        self.atten_lim = atten_lim
        self.dfn = DeepFilterNetStreamingONNX(atten_lim=self.atten_lim)
        self.frame_length = self.dfn.frame_length

        self.upsampler = soxr.ResampleStream(
            in_rate=16000, out_rate=48000, num_channels=1, quality=resampling_quality, dtype="float32"
        )
        self.downsampler = soxr.ResampleStream(
            in_rate=48000, out_rate=16000, num_channels=1, quality=resampling_quality, dtype="float32"
        )
        self.input_buffer = np.array([], dtype=np.float32)

        logger.info(
            "[DeepFilterNetAudioFilter] Initialized (ONNX, atten_lim=%.1f, resampling_quality=%s)",
            self.atten_lim,
            resampling_quality,
        )

    async def filter(self, audio: bytes) -> Optional[bytes]:
        # STAGE 1: int16 -> float32 -> upsample 16kHz -> 48kHz
        audio_int16 = np.frombuffer(audio, dtype=np.int16)
        audio_f32_16k = audio_int16.astype(np.float32) / 32768.0
        audio_float32 = self.upsampler.resample_chunk(audio_f32_16k)

        # Buffer audio for denoising
        self.input_buffer = np.concatenate((self.input_buffer, audio_float32))

        # STAGE 2: Denoise exact frame chunks
        cleaned_frames = []
        while len(self.input_buffer) >= self.frame_length:
            frame = self.input_buffer[:self.frame_length]
            self.input_buffer = self.input_buffer[self.frame_length:]

            clean_frame, _ = self.dfn.process_frame(frame)
            cleaned_frames.append(clean_frame)

        if not cleaned_frames:
            return None

        # STAGE 3: Downsample in float32, quantize last
        clean_f32_48k = np.concatenate(cleaned_frames)
        clean_f32_16k = self.downsampler.resample_chunk(clean_f32_48k)
        clean_int16 = np.clip(clean_f32_16k * 32768.0, -32768, 32767).astype(np.int16)

        return clean_int16.tobytes()

    async def cleanup(self) -> None:
        """Release the ONNX session's native resources."""
        if _HAS_DFN:
            self.dfn.close()
