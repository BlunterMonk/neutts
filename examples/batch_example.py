import os
import soundfile as sf
import torch
from neutts import NeuTTS


def main(
    input_file,
    ref_audio_path,
    ref_text,
    backbone,
    backbone_device="cpu",
    codec_device="cpu",
    language=None,
    output_dir="batch_output",
):
    if not ref_audio_path or not ref_text:
        print("No reference audio or text provided.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("No lines found in input file.")
        return

    print(f"Found {len(lines)} lines to synthesize.")

    os.makedirs(output_dir, exist_ok=True)

    tts = NeuTTS(
        backbone_repo=backbone,
        backbone_device=backbone_device,
        codec_repo="neuphonic/neucodec",
        codec_device=codec_device,
        language=language,
    )

    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    if not os.path.exists(ref_audio_path.replace(".wav", ".pt")):
        print("Encoding reference audio")
        ref_codes = tts.encode_reference(ref_audio_path)
        torch.save(ref_codes, ref_audio_path.replace(".wav", ".pt"))
    else:
        print("Loading pre-encoded reference audio")
        ref_codes = torch.load(ref_audio_path.replace(".wav", ".pt"))

    for i, line in enumerate(lines):
        output_path = os.path.join(output_dir, f"{i:04d}.wav")
        print(f"[{i + 1}/{len(lines)}] Generating: {line[:80]}...")
        wav = tts.infer(line, ref_codes, ref_text)
        sf.write(output_path, wav, 24000)
        print(f"  Saved to {output_path}")

    print(f"Done. {len(lines)} files written to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS Batch Synthesis")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to a text file with one utterance per line",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        default="./samples/jo.wav",
        help="Path to reference audio file",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="./samples/jo.txt",
        help="Reference text corresponding to the reference audio",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_output",
        help="Directory to save output audio files",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="neuphonic/neutts-nano",
        help="Huggingface repo or local path to the backbone checkpoint",
    )
    parser.add_argument(
        "--backbone_device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device for backbone inference",
    )
    parser.add_argument(
        "--codec_device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device for codec inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="eSpeak language code (e.g. 'en', 'fr'). Required when using a local .gguf file path.",
    )
    args = parser.parse_args()
    main(
        input_file=args.input_file,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        backbone=args.backbone,
        backbone_device=args.backbone_device,
        codec_device=args.codec_device,
        language=args.language,
        output_dir=args.output_dir,
    )
