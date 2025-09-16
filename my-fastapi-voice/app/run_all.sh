#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Usage: run_all.sh /abs/path/to/input_audio /abs/path/to/output_dir
IN_AUDIO="${1:-}"; OUT_DIR="${2:-}"
if [[ -z "$IN_AUDIO" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 INPUT_AUDIO_FILE OUT_DIR" >&2
  exit 2
fi

IN_AUDIO="$(readlink -f "$IN_AUDIO")"
OUT_DIR="$(readlink -f "$OUT_DIR")"
mkdir -p "$OUT_DIR"

log(){ printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

# Where this script lives (ollama_transcript_merge.py sits next to it)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
MERGE_PY="$SCRIPT_DIR/ollama_transcript_merge.py"
# Allow override via env; default to your model
OLLAMA_MODEL="${OLLAMA_MODEL:-alibayram/medgemma:27b}"

# ---- Project roots (adjust if different) ----
CV_DIR="/home/user01/clear-voice"
FD_DIR="/home/user01/facebook-denoiser"
DF_DIR="/home/user01/deepfilternet"
FW_DIR="/home/user01/faster-whisper"
SE_DIR="/home/user01/facebook-seamless"

# ---- Python interpreters (prefer venv pythons if present) ----
CV_PY="$CV_DIR/venv/bin/python"; [[ -x "$CV_PY" ]] || CV_PY="python3"
FD_PY="$FD_DIR/facebook-denoiser-venv/bin/python"; [[ -x "$FD_PY" ]] || FD_PY="python3"
DF_PY="$DF_DIR/deepfilternet_venv/bin/python"; [[ -x "$DF_PY" ]] || DF_PY="python3"
FW_PY="$FW_DIR/faster_whisper_venv/bin/python"; [[ -x "$FW_PY" ]] || FW_PY="python3"
SE_PY="$SE_DIR/facebook-seamless/bin/python"; [[ -x "$SE_PY" ]] || SE_PY="python3"

# ---- Job-scoped folders (all inside OUT_DIR) ----
IN_DIR="$OUT_DIR/input"
CV_OUT="$OUT_DIR/clear-voice"
FD_OUT="$OUT_DIR/facebook-denoiser"
DF_OUT="$OUT_DIR/deepfilternet"
FW_OUT="$OUT_DIR/faster-whisper"
SE_OUT="$OUT_DIR/facebook-seamless"
TXT_OUT="$OUT_DIR/texts"

mkdir -p "$IN_DIR" "$CV_OUT" "$FD_OUT" "$DF_OUT" "$FW_OUT" "$SE_OUT" "$TXT_OUT"

# Put the single input inside a folder (some tools expect a directory)
cp -f "$IN_AUDIO" "$IN_DIR/$(basename "$IN_AUDIO")"

# ------------------ PIPELINE ------------------

# ---- job 1: clear-voice ----
log "Job 1: clear-voice"
( cd "$CV_DIR" && "$CV_PY" main.py --input "$IN_DIR" --output "$CV_OUT" --profile hifi_48k )

# ---- job 2: facebook-denoiser ----
log "Job 2: facebook-denoiser"
( cd "$FD_DIR" && "$FD_PY" main.py --input-dir "$IN_DIR" --output-dir "$FD_OUT" )

# ---- job 3: deepfilternet ----
log "Job 3: deepfilternet"
( cd "$DF_DIR" && "$DF_PY" batch_denoise_dfnet.py "$IN_DIR" -o "$DF_OUT" --recursive --model DeepFilterNet3 --post-filter --exts .wav,.mp3,.flac,.ogg,.opus,.m4a,.aac,.wma,.webm,.mp4,.mov,.mkv,.3gp,.caf,.aiff,.aif,.aifc --overwrite )

# ---- job 4: faster-whisper ----
log "Job 4: faster-whisper"
(
  cd "$FW_DIR"
  export LD_LIBRARY_PATH="$FW_DIR/faster_whisper_venv/lib/python3.10/site-packages/nvidia/cublas/lib:$FW_DIR/faster_whisper_venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
  "$FW_PY" main.py --input "$CV_OUT" --output "$FW_OUT" --device cuda --compute-type float16
  "$FW_PY" main.py --input "$FD_OUT" --output "$FW_OUT" --device cuda --compute-type float16
  "$FW_PY" main.py --input "$DF_OUT" --output "$FW_OUT" --device cuda --compute-type float16
)

# ---- job 5: facebook-seamless ----
log "Job 5: facebook-seamless"
(
  cd "$SE_DIR"
  "$SE_PY" main.py "$CV_OUT" "$SE_OUT"
  "$SE_PY" main.py "$FD_OUT" "$SE_OUT"
  "$SE_PY" main.py "$DF_OUT" "$SE_OUT"
)

# ------------------ HARVEST TEXT OUTPUTS ------------------
# Copy every .txt produced under OUT_DIR into OUT_DIR/texts/, but skip any existing merged_output.txt
log "Collecting *.txt results into $TXT_OUT"
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  parent="$(basename "$(dirname "$f")")"
  cp -f "$f" "$TXT_OUT/${parent}__${base}"
done < <(find "$OUT_DIR" -type f -name '*.txt' -not -path "$OUT_DIR/texts/*" -not -name 'merged_output.txt' -print0)

# ------------------ JOB 6: OLLAMA MERGE ------------------
log "Job 6: ollama transcript merge (model: $OLLAMA_MODEL)"
if [[ ! -f "$MERGE_PY" ]]; then
  log "Merge script not found at $MERGE_PY"
  exit 1
fi

# We'll write merged_output.txt inside $TXT_OUT and ONLY handle that file afterward
(
  shopt -s nullglob
  txts=( "$TXT_OUT"/*.txt )
  shopt -u nullglob

  if (( ${#txts[@]} == 0 )); then
    log "No .txt files in $TXT_OUT; skipping merge"
    exit 0
  fi

  cd "$TXT_OUT"
  # If your Python script already writes merged_output.txt by default, no -o is required; keeping it explicit is harmless:
  python3 "$MERGE_PY" -m "$OLLAMA_MODEL" -o "$TXT_OUT/merged_output.txt" --use-chat "${txts[@]}"

  # Confirm model responded and file is non-empty
  if [[ -s "$TXT_OUT/merged_output.txt" ]]; then
    log "Merge successful: $TXT_OUT/merged_output.txt"
  else
    log "Merge script finished but merged_output.txt is missing or empty in $TXT_OUT"
    exit 1
  fi
)

# ------------------ MOVE ONLY THE FINAL MERGED FILE ------------------
# "Corresponding voice folder" = folder that contains the input audio file.
VOICE_DIR="$(dirname "$IN_AUDIO")"
DEST_PATH="$VOICE_DIR/merged_output.txt"

log "Moving ONLY the merged_output.txt to voice folder: $DEST_PATH"
mv -f "$TXT_OUT/merged_output.txt" "$DEST_PATH"

log "Done."




#!/usr/bin/env bash
# stops on the first error; delete 'set -e' if you prefer to keep going
# set -Eeuo pipefail
# 
# # ---- job 1 ----
# ( cd /home/user01/clear-voice && python3 main.py --input /home/user01/my-fastapi-voice/app/uploads --output /home/user01/clear-voice/results --profile hifi_48k )
# 
# # ---- job 2 ----
# ( cd /home/user01/facebook-denoiser && source facebook-denoiser-venv/bin/activate && python main.py --input-dir /home/user01/my-fastapi-voice/app/uploads --output-dir /home/user01/facebook-denoiser/results )
# 
# # ---- job 3 ----
# ( cd /home/user01/deepfilternet && source deepfilternet_venv/bin/activate && python python batch_denoise_dfnet.py /home/user01/my-fastapi-voice/app/uploads -o /home/user01/deepfilternet/results --recursive --model DeepFilterNet3 --post-filter --exts .wav,.mp3,.flac,.ogg,.opus,.m4a,.aac,.wma,.webm,.mp4,.mov,.mkv,.3gp,.caf,.aiff,.aif,.aifc --overwrite )
# 
# # ---- job 4 ----
# ( cd /home/user01/faster-whisper && source faster_whisper_venv/bin/activate && export LD_LIBRARY_PATH="/home/user01/faster-whisper/faster_whisper_venv/lib/python3.10/site-packages/nvidia/cublas/lib/:/home/user01/faster-whisper/faster_whisper_venv/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH" && python main.py --input /home/user01/clear-voice/results --output ./results --device cuda --compute-type float16 && python main.py --input /home/user01/facebook-denoiser/results --output ./results --device cuda --compute-type float16 && python main.py --input /home/user01/deepfilternet/results --output ./results --device cuda --compute-type float16 )
# 
# # ---- job 5 ----
# ( cd /home/user01/facebook-seamless && source facebook-seamless/bin/activate && python main.py /home/user01/clear-voice/results /home/user01/facebook-seamless/results && python main.py /home/user01/facebook-denoiser/results /home/user01/facebook-seamless/results && python main.py /home/user01/deepfilternet/results /home/user01/facebook-seamless/results )

# ---- job 5 ----
# cd /home/user01/my-fastapi-voice
# source clear_voice_venv/bin/activate
# uvicorn app.main:app --host 127.0.0.1 --port 8000
# ssh -R 80:localhost:8000 nokey@localhost.run
# deactivate
# ...repeat the same 4 lines for all 20 commands...
