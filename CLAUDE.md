# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuTTS is a Python package for on-device text-to-speech with instant voice cloning, built by Neuphonic. It uses a two-stage pipeline: a language model backbone generates speech tokens, then an audio codec decodes them to waveforms. Models are hosted on HuggingFace under the `neuphonic/` namespace.

## Build & Install

The build system uses **scikit-build-core** with CMake. CMake compiles and bundles espeak-ng (phonemizer) from source as part of the wheel.

```bash
pip install -e .              # Editable install (builds espeak-ng via CMake)
pip install -e .[all]         # Include llama-cpp-python + onnxruntime
pip install -e .[llama]       # GGUF model support only
pip install -e .[onnx]        # ONNX codec decoder only
```

Python >=3.10, <3.14 required.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/                       # Quick tests (neutts-air + q4-gguf backbones)
RUN_SLOW=true pytest tests/         # All backbones (downloads many models)
pytest tests/test_neutts.py::test_model_loading_and_inference  # Single test group
```

Tests are parametrized over backbone x codec combinations. Quick tests use `neuphonic/neutts-air` and `neuphonic/neutts-air-q4-gguf`. Tests require model downloads from HuggingFace on first run. Streaming tests only run against GGUF backbones.

## Code Formatting & Linting

Pre-commit hooks enforce style. Set up with:
```bash
pip install pre-commit && pre-commit install
```

- **Black**: line length 100
- **Flake8**: line length 100, ignores E203 and W503

## Architecture

### Two-Stage Pipeline (`neutts/neutts.py`)

1. **Backbone (Language Model)** - generates speech tokens from phonemized text + reference speaker codes
   - **Torch backend**: loads via `transformers.AutoModelForCausalLM` (full-precision models)
   - **GGML backend**: loads via `llama-cpp-python` (GGUF quantized models)
2. **Codec** - decodes speech tokens to 24kHz audio waveforms
   - `NeuCodec` / `DistillNeuCodec` (torch) or `NeuCodecOnnxDecoder` (CPU-only ONNX)

### Key Entry Points

- `NeuTTS.infer()` - standard (non-streaming) synthesis
- `NeuTTS.infer_stream()` - streaming synthesis (GGML backend only), yields audio chunks using overlapping window decoding (50-token lookback, 25-token stride)
- `NeuTTS.encode_reference()` - encode reference audio for voice cloning

### Phonemizer (`neutts/phonemizers.py`)

Text is converted to phonemes via bundled espeak-ng. `BasePhonemizer` handles most languages; `FrenchPhonemizer` adds French-specific processing. Language is auto-detected from `BACKBONE_LANGUAGE_MAP` based on the selected backbone.

### NeuTTSAir (`neuttsair/`)

Thin subclass wrapper around NeuTTS with no additional functionality.

### Audio Constants

- Sample rate: 24,000 Hz
- Context window: 2048 tokens (~30s of audio including prompt)
- Streaming chunk: 25 frames / 12,000 samples

## Supported Models

Backbones are mapped to languages in `BACKBONE_LANGUAGE_MAP` (in `neutts/neutts.py`). Available in English, French, German, Spanish. Each language has full-precision, Q4 GGUF, and Q8 GGUF variants. Model families: `neutts-air` (~360M params, English only) and `neutts-nano` (~120M params, multilingual).

## Publishing

GitHub Actions workflow (`.github/workflows/publish.yaml`) publishes to PyPI via manual dispatch.
