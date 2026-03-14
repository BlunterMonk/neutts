"""Entry point: python -m neutts_server"""

import uvicorn

from .config import ServerConfig, build_parser
from .engine import TTSEngine
from .server import app, set_engine


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = ServerConfig.from_args(args)

    tts_engine = TTSEngine(config)
    set_engine(tts_engine)

    print(f"Starting NeuTTS server on {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
