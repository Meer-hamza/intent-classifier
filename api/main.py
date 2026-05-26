import json
import time
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

BASE = Path(__file__).parent.parent
MODELS_DIR = BASE / "models"

print("Loading models...")
clf        = joblib.load(MODELS_DIR / "intent_classifier.joblib")
le         = joblib.load(MODELS_DIR / "label_encoder.joblib")
vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")

with open(MODELS_DIR / "metadata.json") as f:
    metadata = json.load(f)

print(f"  Loaded {len(metadata['classes'])} intent classes")
print("  Server ready!")

app = FastAPI(
    title="User Intent Classifier API",
    description="Classifies support messages into intents in real-time",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(3, ge=1, le=8)

class IntentResult(BaseModel):
    intent: str
    confidence: float
    description: str
    route_to: str
    color: str

class ClassifyResponse(BaseModel):
    text: str
    top_intent: IntentResult
    all_intents: list[IntentResult]
    inference_ms: float
    confidence_tier: str

def get_confidence_tier(confidence: float) -> str:
    if confidence >= 0.85: return "high"
    if confidence >= 0.60: return "medium"
    return "low"

def predict_single(text: str, top_k: int = 3) -> ClassifyResponse:
    start = time.perf_counter()
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    sorted_idx = np.argsort(probs)[::-1]
    classes = le.classes_

    all_intents = []
    for idx in sorted_idx:
        intent_name = classes[idx]
        info = metadata["intents"].get(intent_name, {})
        all_intents.append(IntentResult(
            intent=intent_name,
            confidence=round(float(probs[idx]), 4),
            description=info.get("description", ""),
            route_to=info.get("route_to", "General Support"),
            color=info.get("color", "#6b7280"),
        ))

    top = all_intents[0]
    return ClassifyResponse(
        text=text,
        top_intent=top,
        all_intents=all_intents[:top_k],
        inference_ms=round(elapsed_ms, 2),
        confidence_tier=get_confidence_tier(top.confidence),
    )

@app.get("/")
def root():
    return {"name": "Intent Classifier API", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_accuracy": metadata["accuracy"],
        "training_data": metadata.get("training_data", ""),
        "num_intents": len(metadata["classes"]),
    }

@app.get("/intents")
def list_intents():
    return {"count": len(metadata["intents"]), "intents": metadata["intents"]}

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return predict_single(req.text.strip(), top_k=req.top_k)

@app.post("/classify/batch")
def classify_batch(req: dict):
    texts = req.get("texts", [])
    if not texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty.")
    results = [predict_single(t.strip()).model_dump() for t in texts if t.strip()]
    return {"count": len(results), "results": results}