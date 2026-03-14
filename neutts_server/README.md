# NeuTTS Server

WebSocket streaming TTS server wrapping NeuTTS. Keeps the model loaded in GPU memory and serves real-time audio over localhost.

## Quick Start

```bash
pip install fastapi uvicorn[standard] websockets python-multipart

python -m neutts_server --backbone neuphonic/neutts-air-q4-gguf --backbone-device gpu
```

The server binds to `127.0.0.1:9100` by default.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backbone` | *(required)* | HuggingFace repo ID or local `.gguf` path |
| `--backbone-device` | `cpu` | `cpu` or `gpu` |
| `--codec-repo` | `neuphonic/neucodec-onnx-decoder` | Codec model repo or path |
| `--codec-device` | `cpu` | `cpu` or `gpu` |
| `--language` | auto-detected | eSpeak language code (e.g. `en-us`, `fr-fr`). Required for local `.gguf` paths. |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `9100` | Server port |
| `--voices-dir` | `~/.neutts_server/voices/` | Directory for cached voice embeddings |

---

## Endpoints

### `GET /health`

Returns engine status.

**Response:**

```json
{
  "status": "ok",
  "backbone": "neuphonic/neutts-air-q4-gguf",
  "device": "gpu",
  "busy": false
}
```

**Example:**

```bash
curl http://127.0.0.1:9100/health
```

---

### `POST /v1/voices/encode`

Upload a WAV file to encode a reusable voice. The encoded voice is cached to disk in the voices directory.

**Form fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file upload | yes | WAV audio file of the reference speaker |
| `ref_text` | string | no | Transcript of the reference audio (default: `""`) |

**Response:**

```json
{
  "voice_id": "a3f8c2e1b7d94f60"
}
```

**Example:**

```bash
curl -X POST http://127.0.0.1:9100/v1/voices/encode \
  -F "file=@speaker.wav" \
  -F "ref_text=This is a sample of my voice."
```

---

### `GET /v1/voices`

List all cached voices.

**Response:**

```json
{
  "voices": [
    { "voice_id": "a3f8c2e1b7d94f60", "filename": "a3f8c2e1b7d94f60.pt" },
    { "voice_id": "9c2d4e8f1a3b7056", "filename": "9c2d4e8f1a3b7056.pt" }
  ]
}
```

**Example:**

```bash
curl http://127.0.0.1:9100/v1/voices
```

---

### `WS /v1/tts/stream`

WebSocket endpoint for streaming text-to-speech. Audio is streamed as binary frames in real time.

#### Protocol

1. **Client sends** a JSON text frame with the TTS request:

```json
{
  "text": "Hello, this is a streaming test.",
  "voice_id": "a3f8c2e1b7d94f60",
  "ref_text": "This is a sample of my voice."
}
```

2. **Server streams** binary frames. Each frame contains:
   - 4 bytes: little-endian `uint32` sequence number
   - Remaining bytes: `int16` PCM audio @ 24kHz mono

3. **Server sends** a JSON text frame when generation is complete:

```json
{
  "event": "done",
  "chunks": 8,
  "duration_s": 4.032
}
```

4. **Client can send** a cancel event to abort mid-stream:

```json
{
  "event": "cancel"
}
```

#### Error responses

If the engine is busy or the voice is not found, the server sends:

```json
{
  "event": "error",
  "detail": "Engine is busy with another request"
}
```

#### Python client example

```python
import asyncio
import json
import struct
import wave

import websockets


async def stream_tts(text: str, voice_id: str, ref_text: str, output_path: str):
    uri = "ws://127.0.0.1:9100/v1/tts/stream"
    audio_data = bytearray()

    async with websockets.connect(uri) as ws:
        # Send request
        await ws.send(json.dumps({
            "text": text,
            "voice_id": voice_id,
            "ref_text": ref_text,
        }))

        # Receive frames
        async for message in ws:
            if isinstance(message, bytes):
                seq = struct.unpack("<I", message[:4])[0]
                pcm = message[4:]
                audio_data.extend(pcm)
                print(f"Chunk {seq}: {len(pcm)} bytes")
            else:
                event = json.loads(message)
                if event["event"] == "done":
                    print(f"Done: {event['chunks']} chunks, {event['duration_s']}s")
                    break
                elif event["event"] == "error":
                    print(f"Error: {event['detail']}")
                    break

    # Save to WAV
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(24000)
        wf.writeframes(bytes(audio_data))

    print(f"Saved to {output_path}")


asyncio.run(stream_tts(
    text="Hello world, this is a streaming test.",
    voice_id="a3f8c2e1b7d94f60",
    ref_text="This is a sample of my voice.",
    output_path="output.wav",
))
```

#### PyAudio real-time playback example

```python
import asyncio
import json
import struct

import pyaudio
import websockets


async def stream_and_play(text: str, voice_id: str, ref_text: str):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    uri = "ws://127.0.0.1:9100/v1/tts/stream"

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "text": text,
            "voice_id": voice_id,
            "ref_text": ref_text,
        }))

        async for message in ws:
            if isinstance(message, bytes):
                pcm = message[4:]  # skip sequence header
                stream.write(pcm)
            else:
                break  # done or error

    stream.stop_stream()
    stream.close()
    p.terminate()


asyncio.run(stream_and_play(
    text="Hello world, this is a streaming test.",
    voice_id="a3f8c2e1b7d94f60",
    ref_text="This is a sample of my voice.",
))
```
