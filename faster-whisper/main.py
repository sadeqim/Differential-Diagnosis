#!/usr/bin/env python3
"""
Batch transcribe a folder of .wav files with faster-whisper and save BOTH:
  - English transcript  -> <name>.en.txt
  - Persian (Farsi)     -> <name>.fa.txt

How it works:
- Detects the spoken language automatically.
- Creates a source-language transcript.
- Creates an English transcript:
    * If audio is not English, uses Whisper's built-in translate->English.
    * If audio is English, reuses the source transcript.
- Creates a Persian transcript:
    * If audio is Persian, reuses the source transcript.
    * Otherwise, uses an offline EN->FA translator (Helsinki-NLP MarianMT via transformers).
      (If source isn't English, it pivots through the English translation.)

Requirements:
  pip install faster-whisper transformers sentencepiece torch tqdm

Examples:
  CPU:
    python batch_en_fa_transcribe.py --input ./voices --output ./out

  GPU (faster):
    python batch_en_fa_transcribe.py --input ./voices --output ./out \
      --device cuda --compute-type float16
"""

import argparse
import os
import sys
import math
from typing import List, Tuple

from tqdm import tqdm
from faster_whisper import WhisperModel

# --- Translation (Transformers MarianMT) ---
# We'll lazily construct translators only when needed.
from transformers import pipeline

SUPPORTED_EXT = (".wav",)

def collect_wavs(folder: str, recursive: bool = False) -> List[str]:
    files = []
    if recursive:
        for root, _, names in os.walk(folder):
            for n in names:
                if n.lower().endswith(SUPPORTED_EXT):
                    files.append(os.path.join(root, n))
    else:
        for n in sorted(os.listdir(folder)):
            p = os.path.join(folder, n)
            if os.path.isfile(p) and n.lower().endswith(SUPPORTED_EXT):
                files.append(p)
    return files

def chunk_text_by_chars(text: str, max_chars: int = 800) -> List[str]:
    """
    Simple chunker to avoid blowing past model max length.
    Prefers splitting on sentence-ish boundaries.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks, buf = [], []
    cur_len = 0
    import re
    sentences = re.split(r'(?<=[\.\!\?؛؟])\s+', text)  # includes Persian punctuation
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cur_len + len(s) + 1 > max_chars and buf:
            chunks.append(" ".join(buf))
            buf, cur_len = [s], len(s)
        else:
            buf.append(s)
            cur_len += len(s) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks

class EnFaTranslator:
    """
    Wraps two MarianMT pipelines:
      - en->fa  (Helsinki-NLP/opus-mt-en-fa)
      - fa->en  (Helsinki-NLP/opus-mt-fa-en)
    We only instantiate what we actually need.
    """
    def __init__(self, device_str: str = "cpu"):
        # transformers pipeline device: int (GPU id) or -1 for CPU
        self.device = 0 if device_str == "cuda" else -1
        self._en2fa = None
        self._fa2en = None

    def _get_en2fa(self):
        if self._en2fa is None:
            self._en2fa = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fa", device=self.device)
        return self._en2fa

    def _get_fa2en(self):
        if self._fa2en is None:
            self._fa2en = pipeline("translation", model="Helsinki-NLP/opus-mt-fa-en", device=self.device)
        return self._fa2en

    def en_to_fa(self, text: str) -> str:
        chunks = chunk_text_by_chars(text, 800)
        if not chunks:
            return ""
        out = []
        p = self._get_en2fa()
        for c in chunks:
            res = p(c, max_length=512, clean_up_tokenization_spaces=True)
            out.append(res[0]["translation_text"])
        return "\n".join(out).strip()

    def fa_to_en(self, text: str) -> str:
        chunks = chunk_text_by_chars(text, 800)
        if not chunks:
            return ""
        out = []
        p = self._get_fa2en()
        for c in chunks:
            res = p(c, max_length=512, clean_up_tokenization_spaces=True)
            out.append(res[0]["translation_text"])
        return "\n".join(out).strip()

def transcribe_source(model: WhisperModel, audio_path: str, beam_size: int, vad: bool, vad_silence_ms: int):
    vad_params = dict(min_silence_duration_ms=vad_silence_ms) if vad else None
    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters=vad_params,
        language=None,  # auto-detect
        task="transcribe",
        condition_on_previous_text=True,
        temperature=0.0,
    )
    segments = list(segments_gen)
    text = "\n".join([(s.text or "").strip() for s in segments]).strip()
    lang = info.language or "und"
    prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    return text, lang, prob

def translate_to_english_with_whisper(model: WhisperModel, audio_path: str, beam_size: int, vad: bool, vad_silence_ms: int, src_lang: str):
    vad_params = dict(min_silence_duration_ms=vad_silence_ms) if vad else None
    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters=vad_params,
        language=src_lang,  # hint the detected language
        task="translate",
        condition_on_previous_text=True,
        temperature=0.0,
    )
    segments = list(segments_gen)
    text = "\n".join([(s.text or "").strip() for s in segments]).strip()
    return text

def main():
    ap = argparse.ArgumentParser(description="Batch EN+FA transcripts with faster-whisper.")
    ap.add_argument("--input", required=True, help="Folder containing .wav files")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--model", default="medium", help="Whisper model: tiny/base/small/medium/large-v3/distil-large-v3/etc.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device for faster-whisper")
    ap.add_argument("--compute-type", default=None, help="Compute type (e.g., int8, int8_float16, float16, float32). Default: int8 on CPU, float16 on CUDA if available.")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--vad", dest="vad", action="store_true", help="Enable VAD (recommended)")
    ap.add_argument("--no-vad", dest="vad", action="store_false", help="Disable VAD")
    ap.add_argument("--vad-min-silence-ms", type=int, default=500)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--threads", type=int, default=None, help="OMP_NUM_THREADS for CPU")
    ap.set_defaults(vad=True)
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        print(f"[ERR] Input folder not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.output, exist_ok=True)

    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    # Choose compute type if not provided
    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "float16" if args.device == "cuda" else "int8"

    # Load Whisper model once
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    # Translator (uses transformers). We’ll use GPU for translation only if user picked CUDA.
    translator = EnFaTranslator(device_str=args.device)

    wavs = collect_wavs(args.input, recursive=args.recursive)
    if not wavs:
        print("[ERR] No .wav files found.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(wavs)} file(s). Whisper model={args.model} | device={args.device} | compute={compute_type} | beam={args.beam_size}")

    for audio_path in tqdm(wavs, desc="Processing"):
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out_en = os.path.join(args.output, base + ".en.txt")
        out_fa = os.path.join(args.output, base + ".fa.txt")

        try:
            # 1) Source-language transcript + language detection
            src_text, src_lang, lang_prob = transcribe_source(
                model, audio_path, args.beam_size, args.vad, args.vad_min_silence_ms
            )

            # 2) English transcript
            if src_lang == "en":
                en_text = src_text
            else:
                en_text = translate_to_english_with_whisper(
                    model, audio_path, args.beam_size, args.vad, args.vad_min_silence_ms, src_lang
                )

            # 3) Persian transcript
            if src_lang == "fa":
                fa_text = src_text
            else:
                # pivot through English if needed
                if not en_text:
                    en_text = translate_to_english_with_whisper(
                        model, audio_path, args.beam_size, args.vad, args.vad_min_silence_ms, src_lang
                    )
                fa_text = translator.en_to_fa(en_text) if en_text else ""

            # 4) Save
            with open(out_en, "w", encoding="utf-8") as f:
                f.write(en_text.strip() + "\n")
            with open(out_fa, "w", encoding="utf-8") as f:
                f.write(fa_text.strip() + "\n")

            # small summary on stdout
            print(f"\nOK: {audio_path}")
            print(f"  Detected: {src_lang} (p≈{lang_prob:.2f})")
            print(f"  -> {out_en}")
            print(f"  -> {out_fa}")

        except Exception as e:
            print(f"\n[FAIL] {audio_path}: {e}", file=sys.stderr)

    print("Done.")

if __name__ == "__main__":
    main()

# python main.py --input /home/user01/clear-voice/results --output ./results --device cuda --compute-type float16

# find $VIRTUAL_ENV -name "libcudnn_ops.so*" | head -n 10
# (.venv) user01@user01:~/faster-whisper$ find $VIRTUAL_ENV -name "libcublas.so*" | head -n 10
# /home/user01/my-fastapi-voice/.venv/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12
# (.venv) user01@user01:~/faster-whisper$ export LD_LIBRARY_PATH="/home/user01/my-fastapi-voice/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:/home/user01/my-fastapi-voice/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$L
# D_LIBRARY_PATH"
# (.venv) user01@user01:~/faster-whisper$ python - <<'PY'                                                                                                
# import ctypes
# for so in ["libcublas.so.12","libcudnn_ops.so.9"]:
#     try:
#         ctypes.CDLL(so)
#         print("OK:", so)
#     except OSError as e:
#         print("FAIL:", so, e)
# PY
# OK: libcublas.so.12
# OK: libcudnn_ops.so.9

# export LD_LIBRARY_PATH="/home/user01/faster-whisper/faster_whisper_venv/lib/python3.10/site-packages/nvidia/cublas/lib/:/home/user01/faster-whisper/faster_whisper_venv/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH"