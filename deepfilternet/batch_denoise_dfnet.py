#!/usr/bin/env python3
import argparse
import sys
import shutil
import subprocess
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

# Only use these; DON'T import df.io.*
from df.enhance import enhance, init_df

# Default formats (override with --exts)
SUPPORTED_EXTS = {
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma",
    ".webm", ".mp4", ".mov", ".mkv", ".3gp", ".caf", ".aiff", ".aif", ".aifc"
}

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def decode_with_ffmpeg(in_path: Path, target_sr: int) -> Path:
    """Transcode anything to mono target_sr WAV via ffmpeg for safe loading."""
    tmp_wav = in_path.with_suffix(".tmp.df48k.wav")
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(in_path),
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        str(tmp_wav),
    ]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {in_path}:\n{cp.stderr.decode(errors='ignore')}")
    return tmp_wav

def read_audio_cpu(path: Path, target_sr: int):
    """
    Load on CPU only. Try torchaudio first; if it fails, transcode via ffmpeg then reload.
    Returns (waveform [1,T] float32 CPU in [-1,1], orig_sr, tmp_file_or_None)
    """
    tmp_to_cleanup = None
    try:
        wf, sr = torchaudio.load(str(path))
    except Exception:
        if not has_ffmpeg():
            raise
        tmp_to_cleanup = decode_with_ffmpeg(path, target_sr)
        wf, sr = torchaudio.load(str(tmp_to_cleanup))

    # Downmix to mono
    if wf.ndim == 2 and wf.size(0) > 1:
        wf = wf.mean(dim=0, keepdim=True)

    # Resample to model SR on CPU
    if sr != target_sr:
        wf = torchaudio.functional.resample(wf, sr, target_sr)
        sr = target_sr

    wf = torch.clamp(wf.to(torch.float32), -1.0, 1.0)  # CPU float32 [-1,1]
    return wf, sr, tmp_to_cleanup

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def make_out_path(out_root: Path, input_root: Path, file_path: Path, suffix="_df", out_ext=".wav") -> Path:
    """Mirror input folder structure under out_root and add suffix before extension."""
    try:
        rel = file_path.relative_to(input_root)
        rel = rel.with_suffix("")
        out_rel = rel.as_posix() + suffix + out_ext
    except ValueError:
        out_rel = file_path.stem + suffix + out_ext
    return out_root / out_rel

def main():
    parser = argparse.ArgumentParser(description="Batch denoise voice files with DeepFilterNet.")
    parser.add_argument("inputs", nargs="+", help="One or more input directories (or files).")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for enhanced files.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    parser.add_argument("--model", default=None, help="Model to load (e.g., DeepFilterNet3).")
    parser.add_argument("--post-filter", action="store_true", help="Enable DF post-filter.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"], help="Processing device for the model/features.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--exts", default=",".join(sorted(SUPPORTED_EXTS)),
                        help="Comma-separated list of file extensions to include.")
    parser.add_argument("--keep-input-sr", action="store_true",
                        help="Resample output back to each file's original sample rate.")
    args = parser.parse_args()

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Init DFN (it will move model/features to GPU if available)
    model, df_state, _ = init_df(args.model, post_filter=args.post_filter, log_level="INFO")
    df_sr = getattr(df_state, "sr", None)
    df_sr = df_state.sr() if callable(df_sr) else int(df_sr or 48000)

    # Build file list
    wanted_exts = {
        (e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower())
        for e in args.exts.split(",") if e.strip()
    }
    files = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_file():
            if p.suffix.lower() in wanted_exts:
                files.append((p.parent, p))
        elif p.is_dir():
            it = p.rglob("*") if args.recursive else p.glob("*")
            for fp in it:
                if fp.is_file() and fp.suffix.lower() in wanted_exts:
                    files.append((p, fp))
        else:
            print(f"[WARN] Skipping non-existent path: {p}", file=sys.stderr)

    if not files:
        print("No matching input files found.", file=sys.stderr)
        sys.exit(1)

    # Process
    for root, fpath in tqdm(files, desc="Enhancing", unit="file"):
        out_path = make_out_path(out_root, root, fpath)
        if out_path.exists() and not args.overwrite:
            continue

        tmp = None
        try:
            # 1) Load/Resample on CPU
            wf_cpu, orig_sr, tmp = read_audio_cpu(fpath, target_sr=df_sr)

            # 2) IMPORTANT: enhance() expects CPU input because it calls .numpy() internally.
            #    Do NOT move to CUDA before calling enhance().
            with torch.inference_mode():
                enhanced = enhance(model, df_state, wf_cpu)

            # 3) Optional: return to original SR (still CPU)
            out_sr = df_sr
            if args.keep_input_sr and orig_sr != df_sr:
                enhanced = torchaudio.functional.resample(enhanced, df_sr, orig_sr)
                out_sr = int(orig_sr)

            # 4) Save from CPU
            audio_out = torch.clamp(enhanced.detach().cpu(), -1.0, 1.0)
            ensure_dir(out_path)
            torchaudio.save(str(out_path), audio_out, sample_rate=int(out_sr))

        except Exception as e:
            print(f"[ERROR] {fpath} -> {e}", file=sys.stderr)
        finally:
            if isinstance(tmp, Path) and tmp.exists():
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

if __name__ == "__main__":
    main()



# python batch_denoise_dfnet.py /home/user01/my-fastapi-voice/app/uploads -o /home/user01/deepfilternet/results --recursive --model DeepFilterNet3 --post-filter --exts .wav,.mp3,.flac,.ogg,.opus,.m4a,.aac,.wma,.webm,.mp4,.mov,.mkv,.3gp,.caf,.aiff,.aif,.aifc --overwrite