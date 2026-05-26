from pathlib import Path
import subprocess

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

if not (models_dir / "intent_classifier.joblib").exists():
    print("Models not found — training now...")
    subprocess.run(["python", "training/train.py"], check=True)
else:
    print("Models found — skipping training.")