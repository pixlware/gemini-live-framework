# ADR-0002: Frontend Audio Capture and Streaming Strategy

## Status

DEFERRED

## Date

2026-06-21

## Decision Makers

- Architect (approved by Mehul)

## Context and Problem Statement

To enable a real-time conversation, the web client must continuously capture user microphone input, resample the audio to 16kHz Mono Float32, transcode it to 16-bit signed PCM (PCM16) format, and stream it as binary frames over the active WebSocket connection. The browser's native microphone sample rate depends on the operating system and hardware (typically 44.1kHz or 48kHz). Directly streaming unresampled high-frequency audio or raw float32 samples will fail to authenticate or parse correctly on the Gemini backend, which expects exactly 16kHz PCM16 Mono (related to issue #4).

## Decision Drivers

1. **Self-Contained Deployment**: Keep the frontend client easily deployable (ideally a single HTML/JS file) to facilitate effortless developer setup and local testing.
2. **Minimal Latency & Thread Performance**: Audio capturing and processing must not block the browser's main UI thread, ensuring a butter-smooth 60fps UI.
3. **Hardware Independence**: The resampling algorithm must correctly handle any native hardware sample rate (44.1kHz, 48kHz, 96kHz, etc.) down to 16kHz.
4. **Accuracy & Noise Control**: Capture audio cleanly, applying standard AEC (Echo Cancellation) and Noise Suppression via standard Web Audio constraints.

## Considered Options

### Option 1: ScriptProcessorNode (Main Thread Processing)
Use the deprecated `ScriptProcessorNode` to capture audio chunks on the main thread and downsample them in-place.

- **Pros**: Extremely simple to write; self-contained in a single JavaScript block; works in all legacy browsers.
- **Cons**: Runs on the browser's main UI thread. Heavy UI updates (such as real-time transcript streaming) can cause audio packet drops and stuttering.

### Option 2: Multi-file AudioWorklet
Create a separate `recorder-worklet.js` file that implements `AudioWorkletProcessor`. The main thread registers this file to run in a dedicated audio thread.

- **Pros**: High-performance, runs off the main thread; standard-compliant.
- **Cons**: Requires serving an additional static asset. This complicates deployment, routing, and developer setup.

### Option 3: Inline Blob URL AudioWorklet
Write the `AudioWorkletProcessor` code as an inline template string within the main JS code, compile it into a `Blob`, and instantiate it using a dynamic `Blob URL` (`URL.createObjectURL(blob)`).

- **Pros**:
  - High performance of a dedicated audio rendering thread (`AudioWorklet`).
  - Perfect self-containment in a single HTML file — zero configuration or extra asset serving required.
  - Fully standard-compliant.
- **Cons**: Marginally harder to write and debug due to the stringified code block.

## Decision Outcome

Chosen option: **Option 3: Inline Blob URL AudioWorklet**.

This delivers the best of both worlds: maximum technical performance (running on a dedicated background audio thread via Web Audio Worklet) combined with the ultimate deployment simplicity of a single, self-contained HTML/JS dashboard.

---

## Technical Specifications & Algorithms

### 1. Microphone Access & AudioContext Initialization
Secure mic permissions with standard hardware-level Echo Cancellation, Noise Suppression, and Automatic Gain Control:
```javascript
const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1 // Mono
    }
});

const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
const source = audioCtx.createMediaStreamSource(stream);
const hardwareSampleRate = audioCtx.sampleRate; // e.g. 48000
```

### 2. Inline AudioWorklet Registration
Inline processor definition compiled dynamically:
```javascript
const processorCode = `
  class MicrophoneWorklet extends AudioWorkletProcessor {
    process(inputs, outputs, parameters) {
      const input = inputs[0];
      if (input && input.length > 0) {
        // Send the raw Float32 mono channel buffer back to main thread
        this.port.postMessage(input[0]);
      }
      return true;
    }
  }
  registerProcessor('microphone-worklet', MicrophoneWorklet);
`;

const blob = new Blob([processorCode], { type: 'application/javascript' });
const workletUrl = URL.createObjectURL(blob);
await audioCtx.audioWorklet.addModule(workletUrl);

const workletNode = new AudioWorkletNode(audioCtx, 'microphone-worklet');
source.connect(workletNode);
workletNode.connect(audioCtx.destination); // Required to keep worklet alive
```

### 3. Real-time Resampling & Transcoding (Main Thread)
When the worklet posts raw Float32 samples, we accumulate them in an active recording buffer. When the buffer accumulates sufficient samples (e.g. every 2048 samples), we:
1. Resample from `hardwareSampleRate` (e.g., 48000Hz) to `16000Hz`.
2. Convert Float32 (`[-1.0, 1.0]`) to PCM16 Int16 (`[-32768, 32767]`).
3. Stream the raw `ArrayBuffer` bytes over the active WebSocket.

#### Resampling Algorithm (Linear Interpolation)
```javascript
function resampleBuffer(buffer, inputSampleRate, outputSampleRate) {
    if (inputSampleRate === outputSampleRate) return buffer;
    const ratio = inputSampleRate / outputSampleRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
        const index = Math.round(i * ratio);
        result[i] = buffer[index];
    }
    return result;
}
```

#### Transcoding Algorithm (Float32 to PCM16 Int16)
```javascript
function float32ToInt16(float32Buffer) {
    const int16Buffer = new Int16Array(float32Buffer.length);
    for (let i = 0; i < float32Buffer.length; i++) {
        const s = Math.max(-1.0, Math.min(1.0, float32Buffer[i]));
        int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16Buffer.buffer; // raw binary ArrayBuffer
}
```

---

## Implementation Plan

### Affected Files/Directories

- `templates/index.html` (or `static/index.html` or similar frontend path) — Create a single-page HTML client containing the full UI and Web Audio code.
- `routes.py` — Mount a static files router or serve the HTML from a root GET handler `/`.

### Patterns to Follow

- Accumulate input samples in a floating buffer to avoid fragmented tiny packets. Send packets in chunks of ~2048 samples (at 16kHz, this represents a highly responsive ~128ms packet interval).
- Stop sending binary audio frames immediately when not in a valid "Connected" or "Listening" session state.

### Patterns to Avoid

- Do not instantiate multiple `AudioContext` instances. Maintain a single global reference and reuse/resume it when starting a new session.
- Do not store credentials or API keys on the frontend. The client connects strictly to the `/ws/media-stream` backend proxy.

---

## Verification

- [ ] [Functional] Web page requests mic permission on startup or when the "Start" button is clicked.
- [ ] [Functional] Recorded audio bytes are successfully received and logged on the server.
- [ ] [Structural] Mic samples are verified to be exactly 16kHz PCM16 Mono on the wire.
- [ ] [Performance] Resampling and encoding loops consume less than 1% CPU on standard desktop browsers.
