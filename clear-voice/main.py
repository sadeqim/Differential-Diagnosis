#!/usr/bin/env python3
"""
Preprocess audio for Whisper (& friends) using ClearerVoice-Studio.

Profiles:
  - auto_whisper  : pick FRCRN_SE_16K (fast/safe) or MossFormerGAN_SE_16K (noisier clips)
  - asr_safe      : always FRCRN_SE_16K (recommended default for ASR)
  - asr_max       : always MossFormerGAN_SE_16K (heavier denoise; needs GPU for speed)
  - hifi_48k      : MossFormer2_SE_48K (for listening/mastering, not ASR)

Requires:
  pip install clearvoice  # provides `from clearvoice import ClearVoice`
  ffmpeg in PATH
"""

import argparse
import os
import sys
import tempfile
import subprocess
from typing import Optional, Tuple, Dict

import numpy as np
import soundfile as sf

# --- ClearerVoice-Studio API ---
from clearvoice import ClearVoice  # https://github.com/modelscope/ClearerVoice-Studio

SUPPORTED_IN = (".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".opus", ".wma", ".webm", ".aiff", ".ac3")
DEFAULT_SR_16K = 16000

# --------------------
# Utility: run ffmpeg
# --------------------
def ffprobe_sr(path: str) -> Optional[int]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "default=nw=1:nk=1", path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None

def to_wav_mono_sr(src_path: str, target_sr: int = DEFAULT_SR_16K) -> Tuple[str, Optional[str]]:
    """
    Decode any input to mono WAV at target_sr. Returns (wav_path, temp_path_to_cleanup_or_None)
    If input already matches (wav/mono/target_sr), returns it untouched.
    """
    try:
        if src_path.lower().endswith(".wav"):
            # Quick verify SR & channels
            info = sf.info(src_path)
            if info.samplerate == target_sr and info.channels == 1:
                return src_path, None
        # write to temp wav
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-vn", "-ac", "1", "-ar", str(target_sr),
            "-sample_fmt", "s16",  # ASR-friendly
            tmp_wav
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return tmp_wav, tmp_wav
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed on {src_path} -> {e.stderr.decode('utf-8', errors='ignore') if hasattr(e, 'stderr') else e}") from e

# ----------------------------
# Simple noise score estimator
# ----------------------------
def quick_noise_score(wav_path: str, sr: int = DEFAULT_SR_16K) -> float:
    """
    Single-ended, cheap proxy for "noisiness".
    - Load mono audio.
    - Frame into ~20 ms windows; compute RMS per frame.
    - Use 20th vs 80th percentile RMS as noise vs speech proxies.
    Returns a number ~ 0..1+ where higher ~ noisier (lower SNR).
    """
    x, file_sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    # resample if needed (rare; we already force to 16 kHz)
    if file_sr != sr:
        # polyphase via ffmpeg would be cleaner, but keep it simple here:
        import math
        from scipy.signal import resample_poly  # scipy is in clearvoice reqs
        g = math.gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)

    # normalize to -1..1
    x = np.clip(x, -1.0, 1.0)
    frame = int(0.02 * sr)  # 20 ms
    if len(x) < frame * 3:
        frame = max(1, len(x) // 3)
    rms = np.sqrt(np.maximum(1e-12, np.convolve(x * x, np.ones(frame), mode="valid") / frame))
    if len(rms) < 8:
        return 0.3
    p20 = np.percentile(rms, 20)
    p80 = np.percentile(rms, 80)
    if p80 < 1e-9:
        return 0.3
    snr_like = 20.0 * np.log10((p80 + 1e-9) / (p20 + 1e-9))
    # map: low SNR (<=5 dB) -> ~1.0; high SNR (>=15 dB) -> ~0.0
    score = np.clip(1.0 - (snr_like - 5.0) / 10.0, 0.0, 1.0)
    return float(score)

# ----------------------------
# Model selection & CV object
# ----------------------------
def pick_model(profile: str, noise_score: float, sr_in: int, has_gpu: bool) -> str:
    """
    Decide model by profile + rough noise score.
    """
    profile = profile.lower()
    if profile == "asr_safe":
        return "FRCRN_SE_16K"
    if profile == "asr_max":
        return "MossFormerGAN_SE_16K"
    if profile == "hifi_48k":
        return "MossFormer2_SE_48K"
    # auto_whisper:
    # prefer fast/safe FRCRN; switch to GAN when clearly very noisy
    if noise_score >= 0.55 and has_gpu:
        return "MossFormerGAN_SE_16K"
    return "FRCRN_SE_16K"

def build_cv(task: str, model_name: str) -> ClearVoice:
    # ClearVoice auto-downloads checkpoints; keep it simple
    return ClearVoice(task=task, model_names=[model_name])

# ---------------
# Main pipeline
# ---------------
def process_one(input_path: str,
                output_dir: str,
                profile: str,
                whisper_ready: bool,
                keep_temps: bool,
                cache: Dict[str, ClearVoice],
                device_hint: str = "auto") -> Optional[str]:
    """
    Returns output wav path or None on error.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Ensure we have a WAV 16k mono source (safe for ASR & ClearVoice)
    sr_before = ffprobe_sr(input_path) or 0
    wav_path, tmp_to_cleanup = to_wav_mono_sr(input_path, DEFAULT_SR_16K)

    # 2) Estimate noisiness (cheap) for auto model choice
    noise = quick_noise_score(wav_path, DEFAULT_SR_16K)

    # 3) Decide the model
    import torch
    has_gpu = torch.cuda.is_available() and device_hint != "cpu"
    model_name = pick_model(profile, noise, sr_before, has_gpu)

    # 4) Get/construct the ClearVoice runner (cache by model)
    if model_name not in cache:
        cache[model_name] = build_cv(task="speech_enhancement", model_name=model_name)
    cv = cache[model_name]

    # 5) Run enhancement and write output (always WAV)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_name = f"enhanced_{base}.wav" if profile != "hifi_48k" else f"enhanced48k_{base}.wav"
    out_path = os.path.join(output_dir, out_name)

    try:
        enhanced = cv(input_path=wav_path, online_write=False)  # returns numpy array
        # For Whisper, keep 16 kHz; for 48k profile, ClearVoice model already outputs 48 kHz
        cv.write(enhanced, output_path=out_path)
        return out_path
    except Exception as e:
        print(f"[ERR] {input_path} -> {e}", file=sys.stderr)
        return None
    finally:
        if tmp_to_cleanup and not keep_temps:
            try:
                os.remove(tmp_to_cleanup)
            except OSError:
                pass

def main():
    ap = argparse.ArgumentParser(description="ClearerVoice-Studio preprocessing for Whisper/ASR.")
    ap.add_argument("--input", required=True, help="File or directory")
    ap.add_argument("--output", required=True, help="Output directory for enhanced WAVs")
    ap.add_argument("--profile", default="auto_whisper",
                    choices=["auto_whisper", "asr_safe", "asr_max", "hifi_48k"],
                    help="Model/profile to use (see module docstring)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"], help="Force CPU if needed")
    ap.add_argument("--keep-temps", action="store_true", help="Keep intermediate temp WAVs")
    args = ap.parse_args()

    # Optional: force CPU
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Collect files
    in_path = args.input
    files = []
    if os.path.isdir(in_path):
        for f in sorted(os.listdir(in_path)):
            if f.lower().endswith(SUPPORTED_IN):
                files.append(os.path.join(in_path, f))
    elif os.path.isfile(in_path):
        if in_path.lower().endswith(SUPPORTED_IN):
            files = [in_path]
        else:
            print(f"[WARN] Unsupported type: {in_path}", file=sys.stderr)
    else:
        print(f"[ERR] Not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("[ERR] No input audio found.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(files)} file(s). Profile={args.profile}")

    from tqdm import tqdm
    cache: Dict[str, ClearVoice] = {}
    for f in tqdm(files, desc="Enhancing"):
        out = process_one(
            input_path=f,
            output_dir=args.output,
            profile=args.profile,
            whisper_ready=(args.profile != "hifi_48k"),
            keep_temps=args.keep_temps,
            cache=cache,
            device_hint=args.device
        )
        if out:
            print(f"OK: {f} -> {out}")
        else:
            print(f"FAIL: {f}")

    print("Done.")

if __name__ == "__main__":
    main()


# Example: safest choice for ASR/Whisper (recommended)
# python preprocess_for_whisper.py --input /path/in_audio --output /path/out --profile asr_safe

# Let it auto-switch to the stronger GAN denoiser on very noisy clips (GPU recommended)
# python preprocess_for_whisper.py --input /path/in_audio --output /path/out --profile auto_whisper

# Max suppression (GAN) always
# ython preprocess_for_whisper.py --input /path/in_audio --output /path/out --profile asr_max

# Hi-fi 48 kHz enhancement for listening/mastering (not for Whisper)
# python preprocess_for_whisper.py --input /path/in_audio --output /path/out --profile hifi_48k

# python3 main.py --input /home/user01/my-fastapi-voice/app/uploads --output /home/user01/clear-voice/results --profile hifi_48k