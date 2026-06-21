# ADR-0001: Frontend-Backend WebSocket Protocol

## Status

PROPOSED

## Date

2026-06-21

## Decision Makers

- Architect (approved by Mehul)

## Context and Problem Statement

To implement the Gemini Live Voice Bot, we need a bidirectional real-time communication channel between the single-page web frontend and the FastAPI backend. This channel must support streaming low-latency binary audio frames in both directions (client-to-backend user audio, backend-to-client AI audio) alongside rich JSON control/metadata events (voice activity detection signals, dynamic speech transcriptions, turn completions, interruptions, and usage metrics). We need to define a robust, structured, and decoupled protocol for this interaction (related to issues #2, #4, #5, and #6).

## Decision Drivers

1. **Ultra-low latency**: Conversational speech requires processing round-trip audio and state changes in sub-100ms ranges.
2. **Type Safety & Schema Clarity**: Both client and server must agree on message structures and data types to prevent serialization bugs.
3. **No Over-Encoding Overhead**: Avoid encoding binary audio frames as base64 strings, which increases data footprint by ~33% and incurs CPU rendering overhead.
4. **Decoupled Architecture**: Standardized event format allowing seamless extension of additional events or custom payload types.

## Considered Options

### Option 1: Base64-Encoded JSON Messages
Under this option, all messages (including binary audio data) are sent as JSON text frames over WebSocket. Binary audio is base64-encoded and wrapped in a JSON envelope:
```json
{"type": "audio", "data": "base64_string_here..."}
```

- **Pros**: Easy to parse with a single JSON-only handler in both client and server; unified channel serialization.
- **Cons**: Base64 encoding adds ~33% bandwidth overhead and increases latency/CPU overhead on the client (browser JavaScript) and server (FastAPI).

### Option 2: Native WebSocket Framing (Binary & Text Messages)
WebSocket protocol naturally supports two frame types: Text frames (opcode `0x1`) and Binary frames (opcode `0x2`). Under this option:
- **Audio Chunks**: Transmitted as raw binary frames (no JSON wrapping, direct byte transmission).
- **Control & Metadata Events**: Transmitted as JSON text frames.

- **Pros**:
  - Maximum efficiency (zero packaging or encoding overhead for audio frames).
  - Directly compatible with FastAPI's `WebSocket.send_bytes` and `WebSocket.receive_bytes`.
  - Clear structural separation at the network framing layer.
- **Cons**: Both client and server need dispatch logic based on the incoming WebSocket message type (binary array vs. text string).

## Decision Outcome

Chosen option: **Option 2: Native WebSocket Framing (Binary & Text Messages)**.

This option delivers the lowest possible latency and resource consumption. The overhead of base64 processing is completely avoided. FastAPI and standard HTML5 WebSockets natively handle separate text and binary messages, providing a cleaner, faster, and more professional real-time protocol.

---

## Detailed Protocol Contract

### 1. Inbound (Client → Backend)

The client sends two types of frames to `ws://localhost:8000/ws/media-stream`:

| Message Type | Format | Sample Rate / Format | Description |
|--------------|--------|----------------------|-------------|
| **Audio Frame** | Binary | 16kHz PCM16 Mono (Little-Endian) | Raw recorded microphone samples. |
| **Text Message** | Plain Text | UTF-8 String | Direct text input typed by the user (if keyboard interaction is used). |

*Note*: The client does not need to send complex JSON events, as the backend (via Gemini and Orchestrator) automatically detects user activity and turn completions.

### 2. Outbound (Backend → Client)

The backend streams two types of frames to the client:

| Message Type | Format | Payload | Description |
|--------------|--------|---------|-------------|
| **Audio Frame** | Binary | 24kHz PCM16 Mono (Little-Endian) | Raw synthetic AI voice output from Gemini. |
| **Control Message**| JSON | Structured JSON objects (defined below) | Real-time events, VAD, transcripts, and metrics. |

#### JSON Control Message Schemas

All JSON messages have a top-level wrapper with `"type"` and `"data"` fields:

##### A. Transcript Event (`"transcript"`)
Emitted continuously as speech is transcribed.
```json
{
  "type": "transcript",
  "data": {
    "role": "user" | "model",
    "text": "Hello, how can I help you?",
    "final": false,
    "interrupted": false
  }
}
```

##### B. Interruption Event (`"interruption"`)
Emitted immediately when the backend detects the user has interrupted the model.
```json
{
  "type": "interruption",
  "data": {
    "audio_chunks": 42
  }
}
```

##### C. Turn Complete Event (`"turn_complete"`)
Emitted when the AI companion finishes speaking its current turn.
```json
{
  "type": "turn_complete",
  "data": {
    "audio_chunks": 120
  }
}
```

##### D. Voice Activity Event (`"voice_activity"`)
Emitted when voice activity state changes for user or model.
```json
{
  "type": "voice_activity",
  "data": {
    "role": "user" | "model",
    "voice_activity_type": "ACTIVITY_START" | "ACTIVITY_END"
  }
}
```

##### E. Usage Metadata Event (`"usage_metadata"`)
Emitted periodically or at turn completion to update usage statistics.
```json
{
  "type": "usage_metadata",
  "data": {
    "prompt_token_count": 340,
    "response_token_count": 89,
    "total_token_count": 429,
    "thoughts_token_count": 0,
    "tool_use_prompt_token_count": 0
  }
}
```

##### F. General Event (`"event"`)
Used for generic metadata or notifications.
```json
{
  "type": "event",
  "data": {
    "event": "session_initialized",
    "metadata": {
      "session_id": "sess_abc123"
    }
  }
}
```

---

## Implementation Plan

### Affected Files/Directories

- `routes.py` — Implement the `/ws/media-stream` endpoint.
- `docs/adr/` — This ADR and index.

### Dependencies

None. Standard FastAPI WebSockets and Google GenAI SDK are already configured.

### Patterns to Follow

- Use `FastapiTransport(websocket=websocket)` to wrap the FastAPI socket.
- Create a `GeminiLiveSession(config=gemini_config)` with the specified persona instructions.
- Instantiate `Orchestrator(transport=transport, gemini_session=session)` to run the event pipelines.
- Match existing types defined in `gemini_live/models.py`.

### Patterns to Avoid

- Do not parse JSON inside binary receivers; separate handlers using `websocket.receive()` and pattern match as in `BaseTransport`.
- Do not block the event loop in `routes.py` — let the `Orchestrator` run concurrently.

### Configuration Changes

We must ensure the system prompt in Gemini is configured for a concise, warm AI companion. This will be built in the router using `build_gemini_live_config(system_instruction="...")`.

---

## Verification

- [ ] [Functional] WebSocket accepts connections on `/ws/media-stream` and rejects non-WebSocket protocols.
- [ ] [Functional] Sending binary bytes to `/ws/media-stream` streams audio to Gemini.
- [ ] [Functional] Receiving events from backend matches the JSON schemas defined above.
- [ ] [Structural] All model dumps conform to the schemas defined in `gemini_live/models.py`.
