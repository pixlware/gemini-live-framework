# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.0 (2026-06-03)

### Added
- `SessionLogger` for structured, session-scoped logs with `bind()` / `unbind()` context and optional Google Cloud Logging export (`CLOUD_LOGGING_ENABLED`).
- Shared session logger wired through `Orchestrator` into transport, Gemini session, tool handler, transcription, recording, metrics, turn tracker, timers, and audio filters.
- `OrchestratorActions` with `trigger_bot_timeout()` and `TurnTracker.bot_timeout()` for model-idle timeout flows.
- Time-to-first-byte (TTFB) tracking in `MetricTracker`, exposed on turn-state callbacks and session summaries.
- `enable_google_search` option on `build_gemini_live_config()`.
- `DFN_THREAD_LIMIT` setting to limit DeepFilterNet ONNX thread usage.
- Required `name` parameter on `Timer` for identifiable timer logs.
- WebSocket ping interval on the sample `app.py` server (`ws_ping_interval=10.0`).
- `google-cloud-logging` dependency for cloud log export.

### Changed
- Core components migrated from module `logging` to `SessionLogger` with consistent `[Component]` message prefixes.
- `MetricTracker` session summary logs structured metrics (including TTFB) instead of a single formatted string.
- `websockets` logger level capped at INFO during `setup_logging()`.
- Config builder log labels renamed to `[GeminiLiveConfigBuilder]`.

### Changed (breaking)
- `Timer(...)` now requires `name` as the first argument.

### Fixed
- Pyright-friendly DFN import fallbacks when `dfnstream-py` is missing.
- Telemetry `project_id` default to `""` when unset.

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
