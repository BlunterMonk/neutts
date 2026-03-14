import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerConfig:
    backbone: str
    backbone_device: str
    codec_repo: str
    codec_device: str
    language: str | None
    host: str
    port: int
    voices_dir: Path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ServerConfig":
        return cls(
            backbone=args.backbone,
            backbone_device=args.backbone_device,
            codec_repo=args.codec_repo,
            codec_device=args.codec_device,
            language=args.language,
            host=args.host,
            port=args.port,
            voices_dir=Path(args.voices_dir),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NeuTTS WebSocket streaming server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="HuggingFace repo ID or local path to a GGUF backbone",
    )
    parser.add_argument(
        "--backbone-device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device for backbone inference",
    )
    parser.add_argument(
        "--codec-repo",
        type=str,
        default="neuphonic/neucodec-onnx-decoder",
        help="Codec model repo or path",
    )
    parser.add_argument(
        "--codec-device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device for codec inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="eSpeak language code (e.g. 'en-us', 'fr-fr'). Required for local .gguf paths.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address (use 127.0.0.1 for localhost only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9100,
        help="Server port",
    )
    parser.add_argument(
        "--voices-dir",
        type=str,
        default=str(Path.home() / ".neutts_server" / "voices"),
        help="Directory for cached voice embeddings",
    )
    return parser
