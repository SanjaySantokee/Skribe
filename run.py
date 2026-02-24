#!/usr/bin/env python3
"""
One script to install dependencies and start Skribe.
Creates a venv if needed, runs pip install -r requirements.txt, then starts the server.
Usage: python run.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
BIN = VENV / ("Scripts" if os.name == "nt" else "bin")


def main():
    py = BIN / ("python.exe" if os.name == "nt" else "python")
    pip = BIN / "pip"

    # Create venv if missing
    if not py.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True, cwd=ROOT)

    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([str(pip), "install", "-r", str(ROOT / "requirements.txt")], check=True, cwd=ROOT)

    # Start server
    print("Starting Skribe at http://localhost:8000")
    subprocess.run(
        [str(py), "-m", "uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
