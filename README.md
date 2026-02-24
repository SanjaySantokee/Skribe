# Skribe — Audio to Text Transcriber

A small web app that takes an audio file, transcribes it to English using OpenAI’s [Whisper](https://github.com/openai/whisper), and lets you download the result as a `.txt` file.

## Requirements

- **Python 3.8+**
- **ffmpeg** (used by Whisper to read audio)
  - Windows: `winget install ffmpeg` or [ffmpeg.org](https://ffmpeg.org/download.html)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg` (or your package manager)

## Setup

**Quick start (one command):**

```bash
python run.py
```

This creates a virtual environment if needed, installs dependencies, and starts the server. Then open **http://localhost:8000** in your browser. The first run will download the Whisper model (~140 MB+).

**Manual setup:**

1. Create a virtual environment: `python -m venv venv` then activate it (`venv\Scripts\activate` on Windows, `source venv/bin/activate` on macOS/Linux).
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `uvicorn app:app --reload --host 0.0.0.0 --port 8000`
4. Open **http://localhost:8000**

## Using your GPU (e.g. RTX 3080)

If the app shows **CPU** but you have an NVIDIA GPU, PyTorch was likely installed **without CUDA** (CPU-only). The app can't use the GPU until PyTorch is reinstalled with CUDA support.

1. **Check:** In the project folder run: `python check_gpu.py` — it will say whether CUDA is available and print the exact install command if not.
2. **Fix:** With your venv activated, reinstall PyTorch with CUDA (CUDA 12.1 for recent drivers):
   ```bash
   pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
   For CUDA 11.8 use `cu118` instead of `cu121`. Then run `python check_gpu.py` again and restart the app.
3. **Confirm:** The app's device chip should show **CUDA** and your GPU name once PyTorch sees the GPU.

## Usage

1. Open the app in your browser.
2. Drag and drop an audio file (or click to choose). Supported formats: MP3, WAV, M4A, WebM, OGG, FLAC, etc.
3. Click **Transcribe**. Processing may take a minute depending on length and your machine.
4. When done, the text is shown on the page. Click **Download as .txt** to save the transcription.

## Model size and accuracy

The app uses Whisper’s **`base.en`** model by default (English-optimized). In `app.py` you can change `MODEL_NAME` to:

- `tiny.en` / `base.en` — faster, lower accuracy  
- `small.en` — better accuracy, slower  
- `medium.en` / `large-v3` — best quality, more RAM and time  

The `.en` variants are tuned for English. No API keys required; everything runs locally.

## Project layout

- `run.py` — One script to install and start the app  
- `check_gpu.py` — Check if PyTorch sees your GPU (for CUDA setup)  
- `app.py` — FastAPI app, Whisper transcription, static file serving  
- `static/index.html` — Upload UI and .txt download  
- `requirements.txt` — Python dependencies  
