# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (2026-04-22)

Initial release.

### Added
- `Orchestrator` wiring Gemini Live sessions to transports and tool handlers.
- Gemini Live config builder with tuned VAD defaults.
- Transports: `FastapiTransport` (WebSocket) and `ExotelTransport` (telephony).
- Audio filters: `BaseAudioFilter` interface and `DeepFilterNetAudioFilter` denoiser.
- `AudioTranscoder` supporting PCM16 and μ-law at 8, 16, and 24 kHz.
- Transcription, `TextData`, `VoiceActivityData`, and other event payload models.
- Turn tracking with `TurnTracker`, `ConversationState`, and `on_turn_state_change` callback.
- `Timer` with trigger-based scheduling.
- Call recording via `AudioRecorder`.
- `MetricTracker` for per-turn and session-level timing metrics.
