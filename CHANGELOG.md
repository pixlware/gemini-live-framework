# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.5.0 (2026-08-06)

### Changed
- Upgraded all dependencies to latest compatible versions: `google-genai` 1.73.1 → 2.17.0, `fastapi` 0.135.2 → 0.141.1, `uvicorn` 0.42.0 → 0.52.1, `websockets` 16.0 → 16.1.1, `pydantic` 2.12.5 → 2.13.4, `pydantic-settings` 2.13.1 → 2.15.0, `python-multipart` 0.0.22 → 0.0.32, `soxr` 1.0.0 → 1.1.0, `sherpa_onnx` 1.13.3 → 1.13.4, `google-cloud-logging` 3.15.0 → 3.16.2. `numpy` stays at 2.2.6 (capped by `dfnstream-py<2.3`).
- `GeminiLiveSession.connect()` now constructs the client with `enterprise=True` instead of the legacy `vertexai=True` flag.
- Non-blocking tool results are now sent as `FunctionResponse` with the original call ID instead of `send_client_content` context injection; conversation history no longer contains synthetic `[Tool completed]` user turns.
- `interim_message` is now sent to the model **verbatim** as client text — phrase it as a prompt (e.g. `"repeat this sentence: '...'"`). The framework no longer wraps it in a PROCESSING `FunctionResponse`.
- `AudioRecorder` constructor: `filename` and `output_dir` now default to `""` (lazily resolved to a UUID / `".recordings"`); `storage_type` value renamed from `"gcs"` to `"cloud"`; parameter order changed (`logger` moved last).
- `AudioRecorder` storage refactored: `_save_recording()` is now a router delegating to `_save_local()` or `_save_cloud()`. Both paths build WAV in-memory first via `_build_wav_bytes()`. `_upload_to_gcs()` renamed to `_save_cloud()` (overridable for other providers). Lazy imports (`io`, `google.cloud.storage`) moved to module level.

### Added
- Optional `GEMINI_API_KEY` setting for Vertex AI express-mode authentication. When set, the live session authenticates with the API key (`x-goog-api-key`) instead of ADC while still using `GEMINI_LOCATION` for regional endpoint routing. Gemini Developer keys are not accepted.
- Interim input transcription support: `TranscriptData.interim` flag populated from `LiveServerContent.interim_input_transcription`. Interim chunks are forwarded to the transport for real-time display but skipped by `MetricTracker` and `Transcription` history.
- `VoiceActivityData.audio_offset` populated from server-side `VoiceActivity.audio_offset` (duration string relative to the start of the audio stream).
- Docs for passing through newer `LiveConnectConfig` fields (`translation_config`, `custom_vocabulary` / `adaptation_phrases` via `AudioTranscriptionConfig`) and for `UsageMetadata.service_tier`.
- `scheduling` option on `@tool` / `ToolConfig` (`types.FunctionResponseScheduling`; default `WHEN_IDLE`). Attached to the emitted `ToolHandlerResult` for non-blocking tools only and logged at every stage (handler, orchestrator, session); blocking tools carry no policy.
- `GeminiLiveSession.send_tool_response()` accepts an optional `scheduling` parameter.

### Removed (breaking)
- `GeminiLiveSession.send_interim_tool_response()` and `GeminiLiveSession.send_tool_result_as_context()` — replaced by the verbatim text nudge and the scheduled `FunctionResponse`.
- `ToolResponseAction.SEND_CONTEXT` — non-blocking completions now emit `SEND_RESPONSE` with `scheduling` set.
- `execution_delay` option on `@tool` / `ToolConfig` — it existed only to let the old interim PROCESSING response land before the tool ran. Ordering is now guaranteed (the interim text is awaited before the background task starts) and announcement pacing is owned by `scheduling`; tools that want an artificial delay can `await asyncio.sleep()` themselves.

## 0.4.1 (2026-07-31)

### Changed
- `EventData` model struct renamed field `metadata` to `data` (`data: dict = {}`).

## 0.4.0 (2026-07-31)

### Added
- Batched tool response support via `GeminiLiveSession.send_tool_response_batch()` and turn-level response buffering in `Orchestrator`. Multi-tool function calls in a single turn are aggregated and delivered to Gemini as a single batched `FunctionResponse` upon turn completion.
- `TurnTracker.await_user()` method to transition to `WAITING_FOR_USER` state immediately after connection when `initial_text` is empty, arming the user idle timer without waiting for model audio.
- Automatic `.env` loading in `config.py` using `dotenv.load_dotenv()`.

### Changed
- `BaseToolHandler` duplicate tool call handling now returns an explicit error payload (`{"success": False, "error": "Duplicate tool call detected"}`) in the `FunctionResponse` instead of `None`.
- `app.py` logging setup updated to call `gemini_live.logger.setup_logging()`.
- Tuned default `SileroVad` thresholds (`prefix_padding_ms=512`, `min_speech_duration_ms=192`).

## 0.3.2 (2026-06-18)

### Fixed
- Duplicate tool calls now send an empty `SEND_RESPONSE` back to Gemini instead of being silently dropped, so the model is no longer left waiting on a `FunctionResponse` for the skipped call.

### Changed
- Tool and Gemini session logs now include the tool name in the message and carry `tool_args` / `tool_response` payloads for easier tracing of tool-call flows.

## 0.3.1 (2026-06-17)

### Added
- `silero_vad` injection param on `GeminiLiveSession` so callers can pass a pre-configured `SileroVad` (e.g. tuned `silence_duration_ms`, `threshold`) instead of the hardcoded defaults; defaults unchanged when omitted.

## 0.3.0 (2026-06-16)

### Added
- Local Silero VAD via `sherpa-onnx` (`SileroVad`, bundled `models/silero_vad.onnx`). New `vad_type` (`"gemini"` | `"silero"`) on `build_gemini_live_config()` and `GeminiLiveSession`; in `"silero"` mode audio is gated client-side and manual activity signals plus instant `voice_activity` events are emitted.
- GCS storage for `AudioRecorder` via `storage_type="gcs"`, `bucket_name`, and the `GCS_BUCKET_NAME` setting; custom `filename` parameter.

### Changed
- `build_gemini_live_config(vad_enabled=...)` replaced by `vad_type="gemini"|"silero"`.
- `GeminiLiveSession` now drains responses through a background task and queue with a public `receive()` consumer; `disconnect()` cancels the drain task.
- `Orchestrator.start()` races the Gemini connect against the transport and aborts startup if the client disconnects first; recording finalization is fire-and-forget.
- `AudioRecorder` now writes a stereo WAV (left = user, right = model) at 16 kHz instead of mixed-mono at 24 kHz.

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
