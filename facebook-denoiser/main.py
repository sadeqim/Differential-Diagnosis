import os
import pathlib
import zipfile
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
import soundfile as sf
from tqdm.auto import tqdm
import torchaudio
import subprocess
import uuid
import argparse
import shutil

# ---- NEW: parse input/output directories ----
parser = argparse.ArgumentParser(description="Batch denoise audio files with FacebookResearch Denoiser.")
parser.add_argument("--input-dir", required=True, help="Folder containing input audio files")
parser.add_argument("--output-dir", required=True, help="Folder where denoised WAVs will be written")
args = parser.parse_args()

INPUT_DIR = os.path.abspath(args.input_dir)
OUTPUT_DIR = os.path.abspath(args.output_dir)

# Base dir used only for optional converted-wav cache + final zip location
base_dir = os.path.dirname(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

SAVE_CONVERTED_WAVS = False
CONVERTED_DIR = os.path.join(base_dir, "voices_converted_wav")
if SAVE_CONVERTED_WAVS:
    os.makedirs(CONVERTED_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = pretrained.dns64().to(device).eval()
target_sr = model.sample_rate
target_ch = model.chin

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus", ".aiff", ".aif", ".webm"}

def ffmpeg_convert_to_wav(src: pathlib.Path, dst_wav: pathlib.Path) -> None:
    """Convert any audio to WAV via ffmpeg (keeps original SR/ch)."""
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-f", "wav",
        str(dst_wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

def load_audio_any(path: pathlib.Path) -> tuple[torch.Tensor, int]:
    """
    Try torchaudio first; if it fails (e.g., exotic codec), fall back to ffmpeg->temp WAV then load.
    Returns (tensor[C, T], sample_rate)
    """
    try:
        wav, sr = torchaudio.load(str(path))
        return wav, sr
    except Exception:
        tmp = pathlib.Path("/tmp") / f"{uuid.uuid4().hex}.wav"
        ffmpeg_convert_to_wav(path, tmp)
        wav, sr = torchaudio.load(str(tmp))
        try:
            os.remove(tmp)
        except Exception:
            pass
        return wav, sr

def save_wav(path: pathlib.Path, audio: torch.Tensor, sr: int, subtype: str = "PCM_16"):
    """
    Save as 16-bit PCM WAV via soundfile.
    Expects audio tensor [C, T] in [-1, 1].
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    x = audio.detach().cpu()
    if x.dim() == 2:
        x = x.permute(1, 0).numpy()
        if x.shape[1] == 1:
            x = x[:, 0]
    else:
        x = x.squeeze().numpy()
    sf.write(str(path), x, int(sr), format="WAV", subtype=subtype)

input_root = pathlib.Path(INPUT_DIR)
candidates = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

if not candidates:
    print("No audio files found. Check INPUT_DIR and file extensions.")
else:
    print(f"Found {len(candidates)} audio file(s).")
    output_root = pathlib.Path(OUTPUT_DIR)
    errors = []
    for src in tqdm(candidates, desc="Denoising"):
        rel = src.relative_to(input_root)
        out_dir = output_root / rel.parent
        out_path = out_dir / f"{src.stem}_denoised.wav"
        try:
            to_load = src
            if src.suffix.lower() != ".wav":
                if SAVE_CONVERTED_WAVS:
                    conv_path = pathlib.Path(CONVERTED_DIR) / rel.with_suffix(".wav")
                    ffmpeg_convert_to_wav(src, conv_path)
                    to_load = conv_path
            wav, sr = load_audio_any(to_load)
            wav = wav.to(device)
            wav = convert_audio(wav, sr, target_sr, target_ch)
            with torch.no_grad():
                denoised = model(wav.unsqueeze(0))[0].clamp_(-1, 1)
            save_wav(out_path, denoised, target_sr)
        except Exception as e:
            errors.append((str(src), str(e)))
    if errors:
        print(f"\nCompleted with {len(errors)} error(s):")
        for p, msg in errors[:10]:
            print(f"- {p}: {msg[:200]}{'...' if len(msg)>200 else ''}")
        if len(errors) > 10:
            print(f"... and {len(errors)-10} more.")
    else:
        print("\nAll files processed successfully 🎉")
    print(f"Outputs saved under: {OUTPUT_DIR}")

zip_name_dfn = os.path.join(base_dir, "processed_FacebookResearch_Denoiser")
shutil.make_archive(zip_name_dfn, 'zip', OUTPUT_DIR)
print(f"Zipped {OUTPUT_DIR} as {zip_name_dfn}.zip")
print("You can download the zip manually from the server (e.g., via SCP).")

# python main.py --input-dir /home/user01/my-fastapi-voice/app/uploads --output-dir /home/user01/facebook-denoiser/results