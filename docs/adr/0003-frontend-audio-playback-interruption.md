# ADR-0003: Frontend Low-Latency Audio Playback and Interruption Architecture

## Status

PROPOSED

## Date

2026-06-21

## Decision Makers

- Architect (approved by Mehul)

## Context and Problem Statement

For a warm and conversational AI companion, synthetic voice output must be played back with sub-100ms latency. The backend streams output audio chunks as 24kHz PCM16 Mono bytes. The client needs to decode these bytes into Float32 representation and schedule them sequentially for stutter-free, continuous playback using the Web Audio API. Crucially, the system must support immediate interruption: if the user starts speaking while the AI is talking, the backend immediately detects it, stops streaming, and sends a JSON `"interruption"` control event. The client must instantly stop all scheduled audio playback nodes and purge all buffers to avoid double-talk and lag (related to issue #5 and #6).

## Decision Drivers

1. **Stutter-Free Continuity**: Real-time audio streaming from the network can be jittery. Playback must be queued and scheduled sequentially without gaps, pops, or overlap clicks.
2. **Instant Interruption (< 50ms)**: When an interruption event is received, all active and queued audio must stop playing immediately.
3. **Hardware Resampling Independence**: Decoded 24kHz audio must automatically play correctly on any user speaker setup (whether running at 44.1kHz, 48kHz, or other hardware sample rates).
4. **Self-Contained Simplicity**: Implement the scheduling using vanilla Web Audio API primitives, without massive or heavy external audio library dependencies.

## Considered Options

### Option 1: HTML5 `<audio>` tag streaming with MediaSource Extensions (MSE)
Feed received chunks into a `MediaSource` buffer attached to a standard HTML5 audio element.

- **Pros**: Easy to bind; high-level browser-handled buffering.
- **Cons**: High latency (often > 500ms due to browser internal buffer requirements). Very difficult to perform frame-accurate instant clearing and buffer resets during interruptions.

### Option 2: Sequential Web Audio API Buffer Scheduling
Decode each incoming binary frame to Float32, load it into a separate browser `AudioBuffer` running at exactly 24000Hz, and play it using an `AudioBufferSourceNode` scheduled to start exactly when the previous node finishes.

- **Pros**:
  - Extremely low latency (sub-20ms scheduling accuracy).
  - Native upsampling/resampling from 24kHz to hardware output rate is handled automatically by the browser's audio engine.
  - Pinpoint interruption control: keeping a list of scheduled source nodes allows halting all active/queued nodes instantly.
- **Cons**: Requires manually maintaining a timeline clock (`nextPlayTime`) and cleaning up completed nodes.

## Decision Outcome

Chosen option: **Option 2: Sequential Web Audio API Buffer Scheduling**.

This option provides precise, low-level timing and guarantees near-zero latency playback. By scheduling buffer source nodes relative to the `AudioContext.currentTime` clock, we achieve completely seamless, stutter-free playback. Furthermore, keeping references to all active/queued source nodes provides an elegant and foolproof way to implement sub-10ms interruption response times by iterating and calling `.stop()` on all nodes.

---

## Technical Specifications & Algorithms

### 1. Transcoding Algorithm (PCM16 to Float32)
Converts 16-bit signed integer bytes back to Float32 in the range `[-1.0, 1.0]`:
```javascript
function int16ToFloat32(arrayBuffer) {
    const int16Array = new Int16Array(arrayBuffer);
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
    }
    return float32Array;
}
```

### 2. Audio Scheduler & Playback Queue
Maintain a global scheduler state:
```javascript
let nextPlayTime = 0;
let activeSources = [];
const playSampleRate = 24000; // Gemini output rate

function playAudioChunk(float32Data) {
    if (!audioContext || audioContext.state === 'suspended') return;

    // Create an AudioBuffer at exactly 24kHz Mono
    const audioBuffer = audioContext.createBuffer(1, float32Data.length, playSampleRate);
    audioBuffer.copyToChannel(float32Data, 0);

    const sourceNode = audioContext.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(audioContext.destination);

    const currentTime = audioContext.currentTime;
    // If scheduler has fallen behind the clock, catch up
    if (nextPlayTime < currentTime) {
        nextPlayTime = currentTime;
    }

    // Schedule playback
    sourceNode.start(nextPlayTime);
    
    // Track node for interruption purging
    activeSources.push(sourceNode);

    // Remove node reference upon completion
    sourceNode.onended = () => {
        activeSources = activeSources.filter(s => s !== sourceNode);
    };

    // Increment next scheduled start time
    nextPlayTime += audioBuffer.duration;
}
```

### 3. Playback Interruption (Purging the Timeline)
When the WebSocket receives a message with type `"interruption"` (or when the user manually hits "Stop"), execute:
```javascript
function handleInterruption() {
    console.log("Interruption received! Stopping playback immediately...");
    
    // Stop all active and scheduled source nodes
    activeSources.forEach(source => {
        try {
            source.stop();
        } catch (e) {
            // Node might have already finished or not started yet
        }
    });
    
    // Clear active sources array
    activeSources = [];
    
    // Reset play clock
    nextPlayTime = audioContext ? audioContext.currentTime : 0;
}
```

---

## Implementation Plan

### Affected Files/Directories

- `templates/index.html` — Implement the playback queue, Int16-to-Float32 conversion, sequential play, and interruption listeners in JS.

### Patterns to Follow

- Always wrap `sourceNode.stop()` calls in a `try/catch` block, as calling `.stop()` on a node that hasn't started or has already ended throws an error.
- Always check if `audioContext.state === 'suspended'` and resume it when the user interacts with the page (modern browsers require a user interaction gesture to unlock audio).

### Patterns to Avoid

- Do not attempt to append audio bytes to a single continuous `AudioBuffer`. Browser buffers are immutable once written; creating small sequential `AudioBufferSourceNode`s is the correct and performant browser pattern.

---

## Verification

- [ ] [Functional] Received binary chunks are played aloud on the speaker with no noticeable delays or gaps.
- [ ] [Functional] Sending an interruption event instantly cuts off the audio playback (sub-50ms latency).
- [ ] [Structural] Playback does not leak `AudioBufferSourceNode` references in memory.
- [ ] [Performance] Queued playback operates smoothly without audio popping or crackling.
