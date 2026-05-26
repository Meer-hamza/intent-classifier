import os
from pathlib import Path

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

if not (models_dir / "intent_classifier.joblib").exists():
    print("Models not found — training now...")
    import subprocess
    subprocess.run(["python", "training/train_clinc.py"], check=True)
else:
    print("Models found — skipping training.")