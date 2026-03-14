# Examples

### GGUF Backbones

To run the model with `llama-cpp-python` in GGUF format, select a GGUF backbone when intializing the example script.

```bash
python -m examples.basic_example \
  --input_text "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all." \
  --ref_audio ./samples/jo.wav \
  --ref_text ./samples/jo.txt \
  --backbone neuphonic/neutts-nano-q4-gguf
```

### Pre-encode a reference

Reference encoding can be done ahead of time to reduce latency whilst inferencing the model; to pre-encode a reference you only need to provide a reference audio, as in the following script:

```bash
python -m examples.encode_reference \
 --ref_audio  ./samples/jo.wav \
 --output_path ./samples/jo.pt
 ```

Note that `basic_streaming_example.py` requires a pre-encoded reference. `basic_example.py` will encode your reference if a pre-encoding does not exist, and will save and use it in future runs with the same reference.

### Minimal Latency Example

To take advantage of encoding references ahead of time, we have a compiled the codec decoder into an [onnx graph](https://huggingface.co/neuphonic/neucodec-onnx-decoder) that enables inferencing NeuTTS without loading the encoder.
This can be useful for running the model in resource-constrained environments where the encoder may add a large amount of extra latency/memory usage.

To test the decoder, make sure you have installed ```onnxruntime``` and run the following:

```bash
python -m examples.onnx_example \
  --input_text "State-of-the-art Voice AI has been locked behind web APIs for too long. NeuTTS Air is the world’s first super-realistic, on-device, TTS speech language model with instant voice cloning." \
  --ref_codes samples/plat.pt \
  --ref_text samples/plat.txt \
  --backbone_device gpu \
  --backbone "./neuphonic/neutts-air-Q8_0.gguf"
```

### Streaming Support

To stream the model output in chunks, try out the `basic_streaming_example.py` example. For streaming, only the GGUF backbones are currently supported. Ensure you have `llama-cpp-python`, `onnxruntime` and `pyaudio` installed to run this example.

```bash
python -m examples.basic_streaming_example \
  --input_text "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all." \
  --ref_codes samples/jo.pt \
  --ref_text samples/jo.txt \
  --backbone neuphonic/neutts-nano-q4-gguf
```

python -m examples.basic_streaming_example --input_text "State-of-the-art Voice AI has been locked behind web APIs for too long. NeuTTS Air is the world’s first super-realistic, on-device, TTS speech language model with instant voice cloning." --ref_codes samples/plat.pt --ref_text samples/plat.txt --backbone_device gpu --backbone "./neuphonic/neutts-air-Q8_0.gguf" --backbone_device gpu