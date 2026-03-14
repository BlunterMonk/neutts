import asyncio
import json
import queue
import struct
import sys
import threading
import time

import pyaudio
import websockets

VOICE_ID = "plat"
REF_TEXT = "Want to know the origin of my codename? You see, there are different ranks for the Knight Assassins in Kazimierz. There's three Darksteel and two Lazurite assassins above me, but I and I alone bear the rank of Platinum, from now till the day I die. So I figured it'd make a good codename."
URI = "ws://192.168.1.198:9100/v1/tts/stream"


def _playback_worker(audio_q: queue.Queue, pa_stream: pyaudio.Stream):
    """Drains the queue and writes PCM to the audio device."""
    CHUNK = 2048
    while True:
        data = audio_q.get()
        if data is None:
            break
        for i in range(0, len(data), CHUNK):
            pa_stream.write(data[i:i + CHUNK], exception_on_underflow=False)


async def speak(text: str, pa: pyaudio.PyAudio):
    pa_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    audio_q: queue.Queue[bytes | None] = queue.Queue()
    player = threading.Thread(target=_playback_worker, args=(audio_q, pa_stream))
    player.start()

    total_bytes = 0
    t_start = time.perf_counter()

    try:
        async with websockets.connect(URI) as ws:
            await ws.send(json.dumps({
                "text": text,
                "voice_id": VOICE_ID,
                "ref_text": REF_TEXT,
            }))

            async for message in ws:
                if isinstance(message, bytes):
                    seq = struct.unpack("<I", message[:4])[0]
                    pcm = message[4:]
                    total_bytes += len(pcm)
                    samples = len(pcm) // 2
                    chunk_ms = samples / 24000 * 1000
                    elapsed = (time.perf_counter() - t_start) * 1000
                    print(f"  chunk {seq:2d}: {chunk_ms:6.1f}ms audio, received at {elapsed:7.1f}ms")
                    audio_q.put(pcm)
                else:
                    event = json.loads(message)
                    if event["event"] == "error":
                        print(f"  ERROR: {event['detail']}")
                    else:
                        duration = event.get("duration_s", 0)
                        chunks = event.get("chunks", 0)
                        audio_s = total_bytes / 2 / 24000
                        print(f"  done: {chunks} chunks, {audio_s:.2f}s audio, {duration:.3f}s wall")
                    break
    finally:
        audio_q.put(None)  # signal player to stop
        player.join()      # wait for all audio to finish playing
        pa_stream.stop_stream()
        pa_stream.close()


async def main():
    pa = pyaudio.PyAudio()
    print(f"NeuTTS Server REPL (voice={VOICE_ID})")
    print("Type text and press Enter to speak. Ctrl+C to quit.\n")

    try:
        while True:
            text = await asyncio.get_event_loop().run_in_executor(None, lambda: input("> "))
            text = text.strip()
            if not text:
                continue
            await speak(text, pa)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
    finally:
        pa.terminate()


if __name__ == "__main__":
    asyncio.run(main())
