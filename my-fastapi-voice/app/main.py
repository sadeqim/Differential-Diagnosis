from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import os, re, json, subprocess, sys, io

app = FastAPI(title="Voice Uploader")

# CORS: open by default; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR    = Path(__file__).resolve().parent
STATIC_DIR  = BASE_DIR / "static"
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
SCRIPT_PATH = BASE_DIR / "run_all.sh"   # <- Put run_all.sh next to this file

UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

def next_numeric_id(dir_path: Path) -> int:
    """
    Find max numeric prefix and return +1 (e.g., 12.webm -> 13).
    """
    max_id = 0
    for name in os.listdir(dir_path):
        m = re.match(r"^(\d+)", name)
        if m:
            try:
                max_id = max(max_id, int(m.group(1)))
            except ValueError:
                continue
    return max_id + 1

def _write_status(out_dir: Path, status: str, error: str | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updated": datetime.utcnow().isoformat() + "Z"}
    if error:
        payload["error"] = error
    (out_dir / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _kickoff_job(job_id: int, input_path: str, out_dir_str: str):
    """
    This runs AFTER the response returns (BackgroundTasks).
    It executes run_all.sh and writes status + logs in outputs/{job_id}/.
    """
    out_dir = Path(out_dir_str)
    log_path = out_dir / "run.log"
    _write_status(out_dir, "running")

    try:
        if not SCRIPT_PATH.exists():
            _write_status(out_dir, "error", f"Missing script: {SCRIPT_PATH}")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab", buffering=0) as lf:
            # Run the bash file with input and output args.
            proc = subprocess.Popen(
                ["/bin/bash", str(SCRIPT_PATH), input_path, out_dir_str],
                stdout=lf, stderr=subprocess.STDOUT, cwd=str(BASE_DIR)
            )
            rc = proc.wait()

        if rc == 0:
            _write_status(out_dir, "done")
        else:
            _write_status(out_dir, "error", f"Return code {rc}. See run.log.")
    except Exception as e:
        _write_status(out_dir, "error", f"{type(e).__name__}: {e}")

@app.post("/upload")
async def upload_audio(background: BackgroundTasks, audio: UploadFile = File(...)):
    # Only audio
    if not audio.content_type or not audio.content_type.startswith("audio"):
        return JSONResponse({"ok": False, "error": "Only audio files are allowed."}, status_code=400)

    # Extension detection
    if audio.filename and "." in audio.filename:
        ext = audio.filename.rsplit(".", 1)[-1].lower()
    else:
        ct_tail = (audio.content_type.split("/")[-1] if "/" in audio.content_type else "").split(";")[0]
        mapping = {"mpeg": "mp3", "x-wav": "wav", "m4a": "mp4", "x-matroska": "webm"}
        ext = mapping.get(ct_tail, ct_tail or "webm")

    new_id = next_numeric_id(UPLOAD_DIR)
    dest = UPLOAD_DIR / f"{new_id}.{ext}"

    # Save stream in chunks
    with dest.open("wb") as f:
        while True:
            chunk = await audio.read(1 << 20)  # 1MB
            if not chunk:
                break
            f.write(chunk)

    # Prepare output dir & status, then start background job
    job_out_dir = OUTPUTS_DIR / str(new_id)
    _write_status(job_out_dir, "queued")
    background.add_task(_kickoff_job, new_id, str(dest), str(job_out_dir))

    return {
        "ok": True,
        "id": new_id,
        "saved_as": dest.name,
        "status": "started",
        "status_url": f"/status/{new_id}",
        "results_url": f"/results/{new_id}",
        "log_url": f"/log/{new_id}",
    }

@app.get("/status/{job_id}")
def status(job_id: int):
    out_dir = OUTPUTS_DIR / str(job_id)
    status_file = out_dir / "status.json"
    if not status_file.exists():
        return JSONResponse({"ok": False, "status": "unknown"}, status_code=404)
    data = json.loads(status_file.read_text(encoding="utf-8"))
    return {"ok": True, **data}

@app.get("/results/{job_id}")
def results(job_id: int):
    out_dir = OUTPUTS_DIR / str(job_id)
    status_file = out_dir / "status.json"
    if not out_dir.exists():
        return JSONResponse({"ok": False, "error": "Job not found."}, status_code=404)
    if status_file.exists():
        st = json.loads(status_file.read_text(encoding="utf-8")).get("status")
        if st != "done":
            return {"ok": True, "status": st, "files": []}

    # Read *.txt files
    txts = sorted([p for p in out_dir.glob("*.txt") if p.is_file()])
    items = []
    for p in txts:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = "(could not read text)"
        items.append({"name": p.name, "size": p.stat().st_size, "text": text})

    return {"ok": True, "status": "done", "count": len(items), "files": items}

@app.get("/download/{job_id}/{name:path}")
def download_file(job_id: int, name: str):
    file_path = OUTPUTS_DIR / str(job_id) / name
    if not file_path.exists() or not file_path.is_file():
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    return FileResponse(file_path)

@app.get("/log/{job_id}", response_class=PlainTextResponse)
def get_log(job_id: int):
    log_path = OUTPUTS_DIR / str(job_id) / "run.log"
    if not log_path.exists():
        return PlainTextResponse("(no log yet)")
    return PlainTextResponse(log_path.read_text(encoding="utf-8", errors="replace"))

# Run:
# uvicorn app.main:app --host 127.0.0.1 --port 8000
# Tunnel example:
# ssh -R 80:localhost:8000 nokey@localhost.run
# timeout 30m ssh -tt -N -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -R 80:127.0.0.1:8000 nokey@localhost.run -- --output text

