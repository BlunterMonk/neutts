from __future__ import annotations

import asyncio
import hashlib
import queue
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import torch

from neutts import NeuTTS

from .config import ServerConfig


@dataclass
class _StreamJob:
    """Internal job submitted to the worker thread."""

    text: str
    ref_codes: torch.Tensor
    ref_text: str
    response_queue: asyncio.Queue  # items: bytes | None | Exception
    loop: asyncio.AbstractEventLoop
    cancelled: threading.Event = field(default_factory=threading.Event)


class TTSEngine:
    """Wraps NeuTTS with a single-worker thread and async interface."""

    def __init__(self, config: ServerConfig):
        self._config = config
        self._voices_dir = config.voices_dir
        self._voices_dir.mkdir(parents=True, exist_ok=True)

        print("Initializing NeuTTS engine...")
        self._tts = NeuTTS(
            backbone_repo=config.backbone,
            backbone_device=config.backbone_device,
            codec_repo=config.codec_repo,
            codec_device=config.codec_device,
            language=config.language,
        )
        print("NeuTTS engine ready.")

        self._busy = False
        self._current_job: _StreamJob | None = None
        self._job_queue: queue.Queue[_StreamJob | None] = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    @property
    def backbone_name(self) -> str:
        return self._config.backbone

    @property
    def device(self) -> str:
        return self._config.backbone_device

    # ------------------------------------------------------------------
    # Voice management
    # ------------------------------------------------------------------

    def encode_voice(self, wav_path: str | Path) -> tuple[str, torch.Tensor]:
        """Encode reference audio and return (voice_id, codes)."""
        codes = self._tts.encode_reference(str(wav_path))
        # Deterministic ID from the codes tensor
        raw = codes.numpy().tobytes() if isinstance(codes, torch.Tensor) else codes.tobytes()
        voice_id = hashlib.sha256(raw).hexdigest()[:16]
        return voice_id, codes

    def save_voice(self, voice_id: str, codes: torch.Tensor) -> Path:
        path = self._voices_dir / f"{voice_id}.pt"
        torch.save(codes, path)
        return path

    def load_voice(self, voice_id: str) -> torch.Tensor:
        path = self._voices_dir / f"{voice_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Voice '{voice_id}' not found")
        return torch.load(path, weights_only=True)

    def list_voices(self) -> list[dict[str, str]]:
        voices = []
        for p in sorted(self._voices_dir.glob("*.pt")):
            voices.append({"voice_id": p.stem, "filename": p.name})
        return voices

    # ------------------------------------------------------------------
    # Streaming inference
    # ------------------------------------------------------------------

    async def submit_streaming_job(
        self, text: str, voice_id: str, ref_text: str
    ) -> AsyncGenerator[bytes, None]:
        """Submit a TTS job and yield binary audio frames.

        Each frame: 4-byte LE uint32 sequence number + int16 PCM bytes.
        Raises RuntimeError if the engine is already busy.
        """
        if self._busy:
            raise RuntimeError("Engine is busy with another request")

        ref_codes = self.load_voice(voice_id)
        response_queue: asyncio.Queue[bytes | None | Exception] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        job = _StreamJob(
            text=text,
            ref_codes=ref_codes,
            ref_text=ref_text,
            response_queue=response_queue,
            loop=loop,
        )

        # Put job into the thread-safe queue (non-blocking; maxsize=1 so it
        # raises queue.Full if the worker hasn't picked up the previous job)
        self._job_queue.put_nowait(job)

        seq = 0
        while True:
            item = await response_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            header = struct.pack("<I", seq)
            yield header + item
            seq += 1

    def cancel_current_job(self) -> None:
        """Signal the current job to stop early."""
        job = self._current_job
        if job is not None:
            job.cancelled.set()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Runs in a background thread. Pulls jobs and streams results."""
        while True:
            job = self._job_queue.get()  # blocks until a job arrives
            if job is None:
                break
            self._busy = True
            self._current_job = job
            try:
                self._run_stream_job(job)
            except Exception as exc:
                self._put_to_async(job, exc)
            finally:
                self._put_to_async(job, None)  # sentinel
                self._current_job = None
                self._busy = False

    def _run_stream_job(self, job: _StreamJob) -> None:
        for chunk_f32 in self._tts.infer_stream(job.text, job.ref_codes, job.ref_text):
            if job.cancelled.is_set():
                break
            pcm_i16 = (chunk_f32 * 32767).astype(np.int16)
            self._put_to_async(job, pcm_i16.tobytes())

    def _put_to_async(self, job: _StreamJob, item: bytes | None | Exception) -> None:
        job.loop.call_soon_threadsafe(job.response_queue.put_nowait, item)
