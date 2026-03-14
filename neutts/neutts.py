import os
import random
from typing import Generator
from pathlib import Path
import librosa
import numpy as np
import torch
import re
import warnings
from neucodec import NeuCodec, DistillNeuCodec
from transformers import AutoTokenizer, AutoModelForCausalLM
from .phonemizers import BasePhonemizer, CUSTOM_PHONEMIZERS


BACKBONE_LANGUAGE_MAP = {
    # en models
    "neuphonic/neutts-air": "en-us",
    "neuphonic/neutts-air-q4-gguf": "en-us",
    "neuphonic/neutts-air-q8-gguf": "en-us",
    "neuphonic/neutts-nano": "en-us",
    "neuphonic/neutts-nano-q4-gguf": "en-us",
    "neuphonic/neutts-nano-q8-gguf": "en-us",
    # de models
    "neuphonic/neutts-nano-german": "de",
    "neuphonic/neutts-nano-german-q4-gguf": "de",
    "neuphonic/neutts-nano-german-q8-gguf": "de",
    # fr models
    "neuphonic/neutts-nano-french": "fr-fr",
    "neuphonic/neutts-nano-french-q4-gguf": "fr-fr",
    "neuphonic/neutts-nano-french-q8-gguf": "fr-fr",
    # es models
    "neuphonic/neutts-nano-spanish": "es",
    "neuphonic/neutts-nano-spanish-q4-gguf": "es",
    "neuphonic/neutts-nano-spanish-q8-gguf": "es",
}


def _linear_overlap_add(
    frames: list[np.ndarray], stride: int, power: float = 1.0
) -> np.ndarray:
    # original impl --> https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)

    out = np.zeros((*shape, total_size), dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]

        weight = (0.5 - np.abs(t - 0.5)) ** power

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride

    assert sum_weight.min() > 0
    return out / sum_weight


class NeuTTS:

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
        language=None,
    ):

        # Consts
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length
        self.min_tokens_per_word = 12  # conservative floor: ~20 tokens/word at normal pace
        self.min_new_tokens_floor = 50

        # ggml & onnx flags
        self._is_quantized_model = False
        self._is_onnx_codec = False

        # HF tokenizer
        self.tokenizer = None

        # Load text normalizer + phonemizer + models
        self._load_text_normalizer(language, backbone_repo)

        print("Loading phonemizer...")
        self._load_phonemizer(language, backbone_repo)

        self._load_backbone(backbone_repo, backbone_device)

        self._load_codec(codec_repo, codec_device)

        # Load watermarker (optional)
        try:
            import perth

            self.watermarker = perth.PerthImplicitWatermarker()
        except (ImportError, AttributeError, TypeError) as e:
            warnings.warn(
                f"Perth watermarking unavailable: {e}. "
                "Audio will not be watermarked. "
                "Install with: pip install perth>=0.2.0"
            )
            self.watermarker = None

    # espeak language code -> nemo normalizer language code
    _NEMO_LANG_MAP = {
        "en-us": "en",
        "de": "de",
        "fr-fr": "fr",
        "es": "es",
    }

    def _load_text_normalizer(self, language, backbone_repo):
        language = language or BACKBONE_LANGUAGE_MAP.get(backbone_repo)
        nemo_lang = self._NEMO_LANG_MAP.get(language)

        try:
            from nemo_text_processing.text_normalization.normalize import Normalizer

            print(f"Loading text normalizer for '{nemo_lang}'...")
            self.text_normalizer = Normalizer(input_case="cased", lang=nemo_lang)
        except (ImportError, Exception) as e:
            warnings.warn(
                f"Text normalization unavailable: {e}. "
                "Dates, numbers, and abbreviations will not be expanded. "
                "Install with: pip install nemo_text_processing"
            )
            self.text_normalizer = None

    def _normalize_text(self, text: str) -> str:
        if self.text_normalizer is None:
            return text
        return self.text_normalizer.normalize(text)

    def _load_phonemizer(self, language, backbone_repo):
        if not language:
            if BACKBONE_LANGUAGE_MAP.get(backbone_repo) is not None:
                language = BACKBONE_LANGUAGE_MAP[backbone_repo]
            else:
                raise ValueError(
                    "If you aren't using a Neuphonic model, make sure to specify any "
                    "eSpeak language code as the `language` parameter."
                )

        if language in CUSTOM_PHONEMIZERS:
            self.phonemizer = CUSTOM_PHONEMIZERS[language]
        else:
            self.phonemizer = BasePhonemizer(language_code=language)

    def _load_backbone(self, backbone_repo, backbone_device):
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        if backbone_repo.endswith("gguf"):

            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Please install it with:\n"
                    "    pip install llama-cpp-python"
                ) from e

            seed = random.randint(0, 2**32)
            print(f"Using seed {seed}")

            if os.path.isfile(backbone_repo):
                self.backbone = Llama(
                    model_path=backbone_repo,
                    verbose=False,
                    n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                    n_ctx=self.max_context,
                    mlock=True,
                    flash_attn=True if backbone_device == "gpu" else False,
                    seed=seed,
                )
            else:
                self.backbone = Llama.from_pretrained(
                    repo_id=backbone_repo,
                    filename="*.gguf",
                    verbose=False,
                    n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                    n_ctx=self.max_context,
                    mlock=True,
                    flash_attn=True if backbone_device == "gpu" else False,
                    seed=seed,
                )

            self._is_quantized_model = True

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
            self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo).to(
                torch.device(backbone_device)
            )

    def _load_codec(self, codec_repo, codec_device):

        print(f"Loading codec from: {codec_repo} on {codec_device} ...")

        if codec_repo.endswith(".onnx") and os.path.isfile(codec_repo):
            try:
                from neucodec import NeuCodecOnnxDecoder
            except ImportError as e:
                raise ImportError(
                    "Failed to import NeuCodecOnnxDecoder. "
                    "Make sure `neucodec` and `onnxruntime` are installed."
                ) from e

            self.codec = NeuCodecOnnxDecoder(codec_repo)
            self._is_onnx_codec = True

        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/neucodec-onnx-decoder" | "neuphonic/neucodec-onnx-decoder-int8":

                if codec_device != "cpu":
                    raise ValueError("Onnx decoder only currently runs on CPU.")

                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError(
                        "Failed to import the onnx decoder."
                        " Ensure you have onnxruntime installed as well as neucodec >= 0.0.4."
                    ) from e

                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True

            case _:
                raise ValueError(
                    "Invalid codec repo! Must be one of:"
                    " 'neuphonic/neucodec', 'neuphonic/distill-neucodec',"
                    " 'neuphonic/neucodec-onnx-decoder'."
                )

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Strip non-speakable content from text."""
        # Remove fenced code blocks (``` ... ```)
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove inline code (`...`)
        text = re.sub(r"`[^`]+`", "", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove markdown images ![alt](url)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        # Convert markdown links [text](url) to just text
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        # Remove markdown headings (# ## ### etc.)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove markdown bold/italic markers
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove emoji (unicode emoji blocks)
        text = re.sub(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            r"\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
            r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
            "",
            text,
        )
        # Collapse repeated punctuation (!!!, ..., ???) to single
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        # Remove bullet/list markers (-, *, •) at line starts
        text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
        # Remove numbered list markers (1. 2. etc.) at line starts
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _segment_text(text: str) -> list[str]:
        """Split text into sentences at .!? boundaries, preserving punctuation."""
        # Split on sentence-ending punctuation followed by whitespace or end of string.
        # Keeps the delimiter attached to the preceding segment.
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def infer(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str) -> np.ndarray:
        """
        Perform inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Returns:
            np.ndarray: Generated speech waveform.
        """
        text = self._sanitize_text(text)
        text = self._normalize_text(text)
        segments = self._segment_text(text)
        wavs = []

        for segment in segments:
            if self._is_quantized_model:
                output_str = self._infer_ggml(ref_codes, ref_text, segment)
            else:
                prompt_ids = self._apply_chat_template(ref_codes, ref_text, segment)
                output_str = self._infer_torch(prompt_ids)

            wav = self._decode(output_str)
            if self.watermarker is not None:
                wav = self.watermarker.apply_watermark(wav, sample_rate=24_000)
            wavs.append(wav)

        return np.concatenate(wavs)

    def infer_stream(
        self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str
    ) -> Generator[np.ndarray, None, None]:
        """
        Perform streaming inference to generate speech from
            text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Yields:
            np.ndarray: Generated speech waveform.
        """
        if not self._is_quantized_model:
            raise NotImplementedError("Streaming is not implemented for the torch backend!")

        text = self._sanitize_text(text)
        text = self._normalize_text(text)
        segments = self._segment_text(text)
        for segment in segments:
            yield from self._infer_stream_ggml(ref_codes, ref_text, segment)

    def encode_reference(self, ref_audio_path: str | Path):
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def _decode(self, codes: str):

        # Extract speech token IDs using regex
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]

        if len(speech_ids) > 0:

            # Onnx decode
            if self._is_onnx_codec:
                codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
                recon = self.codec.decode_code(codes)

            # Torch decode
            else:
                with torch.no_grad():
                    codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                        self.codec.device
                    )
                    recon = self.codec.decode_code(codes).cpu().numpy()

            return recon[0, 0, :]
        else:
            raise ValueError("No valid speech tokens found in the output.")

    def _to_phones(self, text: str) -> str:
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        phones = " ".join(phones)
        return phones

    def _apply_chat_template(
        self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> list[int]:

        input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)

        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1 :]  # noqa
        )

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)

        return ids

    def _infer_torch(self, prompt_ids: list[int]) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=0.5,
                top_k=50,
                use_cache=True,
                min_new_tokens=50,
            )
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False
        )
        return output_str

    def _get_ggml_end_token_id(self) -> int:
        end_ids = self.backbone.tokenize(b"<|SPEECH_GENERATION_END|>", special=True)
        return end_ids[0]

    def _estimate_min_tokens(self, input_text: str) -> int:
        """Estimate minimum speech tokens from input text word count."""
        word_count = len(input_text.split())
        return max(self.min_new_tokens_floor, word_count * self.min_tokens_per_word)

    def _ggml_token_stream(self, prompt: str, min_tokens: int) -> Generator[str, None, None]:
        """Stream tokens from GGML backend, suppressing the end token for the first N tokens."""
        end_token = "<|SPEECH_GENERATION_END|>"
        end_id = self._get_ggml_end_token_id()

        # Phase 1: generate min tokens with end token suppressed
        accumulated = []
        for item in self.backbone(
            prompt,
            max_tokens=min_tokens,
            logit_bias={str(end_id): -100.0},
            temperature=1.0,
            top_k=50,
            stream=True,
        ):
            text = item["choices"][0]["text"]
            accumulated.append(text)
            yield text

        # Phase 2: continue with stop token allowed
        continued_prompt = prompt + "".join(accumulated)
        for item in self.backbone(
            continued_prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=[end_token],
            stream=True,
        ):
            yield item["choices"][0]["text"]

    def _infer_ggml(self, ref_codes: list[int], ref_text: str, input_text: str) -> str:
        min_tokens = self._estimate_min_tokens(input_text)
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        end_token = "<|SPEECH_GENERATION_END|>"
        end_id = self._get_ggml_end_token_id()

        # Phase 1: generate min tokens with end token suppressed
        output1 = self.backbone(
            prompt,
            max_tokens=min_tokens,
            logit_bias={str(end_id): -100.0},
            temperature=1.0,
            top_k=50,
        )
        text1 = output1["choices"][0]["text"]

        # Phase 2: continue with stop token allowed
        output2 = self.backbone(
            prompt + text1,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=[end_token],
        )
        text2 = output2["choices"][0]["text"]

        return text1 + text2

    def _infer_stream_ggml(
        self, ref_codes: torch.Tensor, ref_text: str, input_text: str
    ) -> Generator[np.ndarray, None, None]:
        raw_input_text = input_text
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        min_tokens = self._estimate_min_tokens(raw_input_text)
        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for output_str in self._ggml_token_stream(prompt, min_tokens):
            token_cache.append(output_str)

            if (
                len(token_cache[n_decoded_tokens:])
                >= self.streaming_frames_per_chunk + self.streaming_lookforward
            ):

                # decode chunk
                tokens_start = max(
                    n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames)
                    * self.hop_length
                )
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = (
                    recon
                    if self.watermarker is None
                    else self.watermarker.apply_watermark(recon, sample_rate=24_000)
                )
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                # postprocess
                processed_recon = _linear_overlap_add(
                    audio_cache, stride=self.streaming_stride_samples
                )
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        # final decoding handled seperately as non-constant chunk size
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if len(token_cache) > n_decoded_tokens:
            tokens_start = max(
                len(token_cache)
                - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens),
                0,
            )
            sample_start = (
                len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames
            ) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = (
                recon
                if self.watermarker is None
                else self.watermarker.apply_watermark(recon, sample_rate=24_000)
            )
            recon = recon[sample_start:]
            audio_cache.append(recon)

            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon
