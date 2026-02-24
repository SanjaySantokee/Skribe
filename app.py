"""
Transcriber web app: upload audio, transcribe with Whisper, download as .txt
"""
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import List

import torch
import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from whisper.audio import N_SAMPLES, SAMPLE_RATE, pad_or_trim

app = FastAPI(title="Skribe Transcriber")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# large-v3-turbo is fast and accurate. Use GPU for best speed (5–10x faster than CPU).
MODEL_NAME = "large-v3-turbo"
model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Job state: job_id -> { progress, status, text?, filename?, error?, duration_seconds? }
jobs: dict = {}
jobs_lock = threading.Lock()

CHUNK_SAMPLES = 30 * SAMPLE_RATE   # 30 seconds per chunk
STEP_SAMPLES = 30 * SAMPLE_RATE   # no overlap = fewest chunks, fastest


@app.on_event("startup")
async def load_model():
    global model
    # Load on GPU if available (much faster); otherwise CPU
    model = whisper.load_model(MODEL_NAME, device=DEVICE)
    # PyTorch 2+ compile can speed up inference (optional)
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass


def _format_sentences(text: str) -> str:
    """Insert two line breaks after each sentence so each has an empty line between."""
    if not text or not text.strip():
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    return "\n\n".join(p.strip() for p in parts if p.strip())


def _merge_overlapping_transcripts(parts: List[str]) -> str:
    """Merge chunk transcripts, removing overlapping duplicate text at boundaries."""
    if not parts:
        return ""
    result = parts[0].strip()
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        result_words = result.split()
        part_words = part.split()
        overlap_len = 0
        for n in range(min(8, len(result_words), len(part_words)), 0, -1):
            if result_words[-n:] == part_words[:n]:
                overlap_len = n
                break
        if overlap_len > 0:
            result += " " + " ".join(part_words[overlap_len:])
        else:
            result += " " + part
    return result.strip()


def _run_transcribe_job(job_id: str, audio_path: Path, filename_stem: str):
    start = time.perf_counter()
    try:
        audio = whisper.load_audio(str(audio_path))
        n_samples = len(audio)
        with jobs_lock:
            jobs[job_id]["progress"] = 1
        if n_samples == 0:
            with jobs_lock:
                jobs[job_id].update(
                    status="error",
                    progress=0,
                    error="Audio file has no content or could not be loaded.",
                )
            return
        chunk_starts = list(range(0, n_samples, STEP_SAMPLES))
        if chunk_starts[-1] + CHUNK_SAMPLES < n_samples:
            chunk_starts.append(max(0, n_samples - CHUNK_SAMPLES))
        n_chunks = len(chunk_starts)
        parts = []
        for i, start_sample in enumerate(chunk_starts):
            end_sample = min(start_sample + CHUNK_SAMPLES, n_samples)
            chunk = audio[start_sample:end_sample]
            chunk = pad_or_trim(chunk, length=N_SAMPLES)
            use_fp16 = DEVICE == "cuda"
            result = model.transcribe(
                chunk,
                language="en",
                fp16=use_fp16,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                temperature=0,
            )
            part = (result.get("text") or "").strip()
            if part:
                parts.append(part)
            progress = int(round((i + 1) / n_chunks * 100))
            with jobs_lock:
                jobs[job_id]["progress"] = min(progress, 100)
        raw = _merge_overlapping_transcripts(parts)
        text = _format_sentences(raw)
        duration_seconds = round(time.perf_counter() - start, 1)
        with jobs_lock:
            jobs[job_id].update(
                status="done",
                progress=100,
                text=text,
                filename=filename_stem,
                duration_seconds=duration_seconds,
            )
    except Exception as e:
        with jobs_lock:
            jobs[job_id].update(
                status="error",
                error=str(e),
            )
    finally:
        audio_path.unlink(missing_ok=True)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    suffix = Path(file.filename).suffix or ".bin"
    if suffix.lower() not in {".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".flac", ".mpeg", ".mpga", ".wma"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Use MP3, WAV, M4A, WebM, OGG, FLAC, or similar.",
        )
    job_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    with jobs_lock:
        jobs[job_id] = {
            "progress": 0,
            "status": "processing",
            "text": None,
            "filename": Path(file.filename).stem,
            "error": None,
            "duration_seconds": None,
        }
    thread = threading.Thread(
        target=_run_transcribe_job,
        args=(job_id, tmp_path, Path(file.filename).stem),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/api/device")
def get_device():
    """Return whether CUDA/GPU is in use and the device name for the UI chip."""
    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else None
    # PyTorch can be installed without CUDA (CPU-only build) - then cuda is False even with a GPU
    cuda_built = torch.version.cuda is not None
    hint = None
    if not cuda and cuda_built:
        hint = "CUDA drivers or GPU issue. Run: python check_gpu.py"
    elif not cuda and not cuda_built:
        hint = "PyTorch is CPU-only. Reinstall with CUDA: see README or run python check_gpu.py"
    return {
        "cuda": cuda,
        "device_name": device_name,
        "cuda_built": cuda_built,
        "hint": hint,
    }


@app.get("/api/transcribe/status/{job_id}")
def get_status(job_id: str):
    with jobs_lock:
        state = jobs.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "progress": state["progress"],
        "status": state["status"],
        "text": state.get("text"),
        "filename": state.get("filename"),
        "error": state.get("error"),
        "duration_seconds": state.get("duration_seconds"),
    }


# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
