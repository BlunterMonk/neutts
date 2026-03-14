# Windows + CUDA: Building NeuTTS from Source

Step-by-step guide for building NeuTTS from source on Windows with NVIDIA GPU (CUDA) support for GGUF model inference.

## Prerequisites

- **Python** >= 3.10, < 3.14
- **Git**
- **NVIDIA GPU** with up-to-date drivers

## 1. Install Build Tools

### Visual Studio C++ Build Tools

NeuTTS builds espeak-ng from source via CMake, which requires a C/C++ compiler.

1. Download [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. In the installer, select the **"Desktop development with C++"** workload
3. Click Install

If Build Tools is already installed but missing the C++ workload, open **Visual Studio Installer**, click **Modify**, and add it.

You can verify the installation by checking that `cl.exe` exists:
```
dir "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
```

### CUDA Toolkit

Required for building llama-cpp-python with GPU support. Install the version that matches your PyTorch CUDA version (e.g., CUDA 12.8 for `cu128`).

```powershell
winget install Nvidia.CUDA --version 12.8
```

Or download from [NVIDIA's CUDA Toolkit page](https://developer.nvidia.com/cuda-toolkit-archive).

Verify with:
```
nvcc --version
```

> **Note:** Close and reopen your terminal after installing the CUDA Toolkit so `nvcc` is on your PATH.

### CMake

Install via pip (used by scikit-build-core to compile espeak-ng):
```bash
pip install cmake scikit-build-core
```

## 2. Install PyTorch with CUDA

The default `pip install torch` pulls a CPU-only build. You must install from PyTorch's CUDA index **before** installing neutts, otherwise neutts's dependencies will pull the CPU version.

For CUDA 12.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For other CUDA versions, replace `cu128` with the appropriate tag (e.g., `cu124`, `cu126`). See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for the full list.

Verify:
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 3. Build and Install NeuTTS from Source

The build compiles espeak-ng via CMake. The CMake generator and compiler must be set so that scikit-build-core can find MSVC.

```bash
git clone https://github.com/neuphonic/neutts.git
cd neutts

set CMAKE_GENERATOR=Visual Studio 17 2022
pip install -e .
```

If using Git Bash or MSYS2:
```bash
export CMAKE_GENERATOR="Visual Studio 17 2022"
pip install -e .
```

## 4. Install llama-cpp-python with CUDA

This compiles llama.cpp's GGML backend with CUDA kernel support. Make sure `nvcc` is on your PATH.

### PowerShell
```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=ON"
$env:CMAKE_GENERATOR = "Visual Studio 17 2022"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Git Bash
```bash
export CMAKE_ARGS="-DGGML_CUDA=ON"
export CMAKE_GENERATOR="Visual Studio 17 2022"
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin:$PATH"
export CUDA_PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

> **Note:** This build takes 10-20 minutes as it compiles CUDA kernels for multiple GPU architectures.

## 5. Fix numpy (if needed)

llama-cpp-python may pull a newer numpy that conflicts with neutts's pinned version. If you see a numpy version warning, fix it:

```bash
pip install "numpy~=2.2.6"
```

## 6. Verify the Full Setup

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')

from llama_cpp import Llama
print('llama-cpp-python: OK')

import neutts
print('neutts: OK')
"
```

Expected output:
```
PyTorch: 2.10.0+cu128, CUDA: True, GPU: NVIDIA GeForce RTX 5080 Laptop GPU
llama-cpp-python: OK
neutts: OK
```

## 7. Run a GGUF Model on GPU

```python
from neutts import NeuTTS
import soundfile as sf

tts = NeuTTS(
    backbone_repo="neuphonic/neutts-air-q4-gguf",
    backbone_device="gpu",
    codec_repo="neuphonic/neucodec",
    codec_device="cpu",
)

ref_text = open("samples/jo.txt").read().strip()
ref_codes = tts.encode_reference("samples/jo.wav")

wav = tts.infer("Hello, this is a test of GPU-accelerated speech synthesis.", ref_codes, ref_text)
sf.write("output.wav", wav, 24000)
```

## Troubleshooting

### `CMake Error: CMAKE_C_COMPILER not set`
The C++ build tools workload is not installed or not found. Make sure:
- "Desktop development with C++" is installed in Visual Studio Installer
- `CMAKE_GENERATOR` is set to `"Visual Studio 17 2022"`

### `pip install torch` replaces CUDA version with CPU version
Always install PyTorch from the CUDA index **first**, then install neutts. If neutts pulls a CPU torch, reinstall:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### `nvcc not found` when building llama-cpp-python
- Ensure the CUDA Toolkit is installed (not just the driver)
- Restart your terminal after installing
- In Git Bash, manually add it to PATH:
  ```bash
  export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin:$PATH"
  ```

### llama-cpp-python build takes forever or fails
- The CUDA build compiles kernels for many GPU architectures. 10-20 minutes is normal.
- If it fails with compiler errors, ensure your CUDA Toolkit version matches your PyTorch CUDA version.
- You can restrict target architectures to speed up the build (e.g., for RTX 50-series/Blackwell only):
  ```bash
  export CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120"
  ```

### numpy version conflicts
Run `pip install "numpy~=2.2.6"` after all other packages are installed to satisfy neutts's pin.
