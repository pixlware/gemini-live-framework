# Architecture Decision Records (ADRs)

This directory contains the Architecture Decision Records for the **Gemini Live Voice Bot** project.

## Index of ADRs

| Number | Title | Status | Date | Summary |
|---|---|---|---|---|
| [ADR-0001](./0001-frontend-backend-websocket-protocol.md) | Frontend-Backend WebSocket Protocol | PROPOSED | 2026-06-21 | Native text & binary WS framing for low latency control/audio streaming. |
| [ADR-0002](./0002-frontend-audio-capture-streaming.md) | Frontend Audio Capture and Streaming Strategy | DEFERRED | 2026-06-21 | Inline Blob URL AudioWorklet for low latency, non-blocking 16kHz PCM16 Mono capture. |
| [ADR-0003](./0003-frontend-audio-playback-interruption.md) | Frontend Low-Latency Audio Playback and Interruption Architecture | DEFERRED | 2026-06-21 | Sequential Web Audio buffer scheduling with active source tracking for instant interruptions. |

## Status Lifecycle

All architectural designs follow the PixlCrew consensus process:
1. **PROPOSED**: Under review by team lead (Mehul).
2. **ACCEPTED**: Approved for implementation.
3. **SUPERSEDED**: Replaced by a newer decision record.
