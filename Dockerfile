FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN pip install --no-cache-dir scikit-build-core \
    && pip wheel --no-cache-dir -w /wheels . \
    && pip wheel --no-cache-dir -w /wheels \
        llama-cpp-python onnxruntime \
        fastapi "uvicorn[standard]" websockets python-multipart

# ---------------------------------------------------------------------------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

RUN mkdir -p /models /voices \
    && curl -L -o /models/neutts-air-Q4_0.gguf \
        https://huggingface.co/neuphonic/neutts-air-q4-gguf/resolve/main/neutts-air-Q4_0.gguf

EXPOSE 9100

ENTRYPOINT ["python", "-m", "neutts_server"]
CMD ["--backbone", "/models/neutts-air-Q4_0.gguf", "--language", "en-us", "--host", "0.0.0.0", "--port", "9100", "--voices-dir", "/voices"]
