from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .engine import TTSEngine
from .protocol import (
    DoneEvent,
    ErrorEvent,
    GenerateResponse,
    HealthResponse,
    TTSRequest,
    VoiceEncodeResponse,
    VoiceInfo,
    VoiceListResponse,
)

app = FastAPI(title="NeuTTS Server")

# Set by __main__.py before uvicorn starts.
engine: TTSEngine | None = None


def set_engine(e: TTSEngine) -> None:
    global engine
    engine = e


# ------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        backbone=engine.backbone_name,
        device=engine.device,
        busy=engine.busy,
    )


@app.post("/v1/voices/encode")
async def encode_voice(
    file: UploadFile,
    ref_text: str = "",
) -> VoiceEncodeResponse:
    """Upload a WAV file and encode it as a reusable voice."""
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        voice_id, codes = engine.encode_voice(tmp_path)
        engine.save_voice(voice_id, codes)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return VoiceEncodeResponse(voice_id=voice_id)


@app.post("/v1/tts/generate")
async def tts_generate(req: TTSRequest) -> StreamingResponse:
    """Run full (non-streaming) TTS and return a WAV file."""
    try:
        t_start = time.perf_counter()
        wav = await engine.submit_generate_job(req.text, req.voice_id, req.ref_text)
        duration_s = round(time.perf_counter() - t_start, 3)

        buf = io.BytesIO()
        sf.write(buf, wav, engine.sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={
                "X-Duration-S": str(duration_s),
                "X-Sample-Rate": str(engine.sample_rate),
            },
        )
    except FileNotFoundError as exc:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": str(exc)})
    except RuntimeError as exc:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/v1/voices")
async def list_voices() -> VoiceListResponse:
    entries = engine.list_voices()
    return VoiceListResponse(voices=[VoiceInfo(**v) for v in entries])


# ------------------------------------------------------------------
# WebSocket streaming endpoint
# ------------------------------------------------------------------


@app.websocket("/v1/tts/stream")
async def tts_stream(ws: WebSocket) -> None:
    await ws.accept()

    try:
        # 1. Receive the TTS request (JSON text frame)
        raw = await ws.receive_text()
        req = TTSRequest.model_validate_json(raw)

        if engine.busy:
            await ws.send_text(ErrorEvent(detail="Engine is busy").model_dump_json())
            await ws.close(code=1013)  # Try Again Later
            return

        # 2. Stream audio chunks
        chunk_count = 0
        t_start = time.perf_counter()

        async for frame in engine.submit_streaming_job(req.text, req.voice_id, req.ref_text):
            await ws.send_bytes(frame)
            chunk_count += 1

        duration_s = round(time.perf_counter() - t_start, 3)

        # 3. Send done event
        done = DoneEvent(chunks=chunk_count, duration_s=duration_s)
        await ws.send_text(done.model_dump_json())

    except FileNotFoundError as exc:
        await ws.send_text(ErrorEvent(detail=str(exc)).model_dump_json())
        await ws.close(code=1008)  # Policy Violation
    except RuntimeError as exc:
        await ws.send_text(ErrorEvent(detail=str(exc)).model_dump_json())
        await ws.close(code=1013)
    except WebSocketDisconnect:
        # Client disconnected — cancel running job if any
        engine.cancel_current_job()
    except Exception as exc:
        try:
            await ws.send_text(ErrorEvent(detail=f"Internal error: {exc}").model_dump_json())
            await ws.close(code=1011)
        except Exception:
            pass
