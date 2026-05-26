import json
import time
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from sentence_transformers import SentenceTransformer
# ── Paths ────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
MODELS_DIR = BASE / "models"

# ── Load Models ──────────────────────────────────────────────
print("Loading models...")
clf       = joblib.load(MODELS_DIR / "intent_classifier.joblib")
le        = joblib.load(MODELS_DIR / "label_encoder.joblib")

embedder = joblib.load(MODELS_DIR / "embedder")

with open(MODELS_DIR / "metadata.json") as f:
    metadata = json.load(f)

print(f"  Loaded {len(metadata['classes'])} intent classes")
print("  Server ready!")

# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="User Intent Classifier API",
    description="Classifies user support messages into intents in real-time",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ──────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, example="I want to cancel my subscription")
    top_k: Optional[int] = Field(3, ge=1, le=8, description="Number of top intents to return")

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
    confidence_tier: str   # "high" | "medium" | "low"

class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)

class HealthResponse(BaseModel):
    status: str
    model_accuracy: float
    num_intents: int
    num_training_samples: int

# ── Helper ───────────────────────────────────────────────────
def get_confidence_tier(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    elif confidence >= 0.60:
        return "medium"
    return "low"

def predict_single(text: str, top_k: int = 3) -> dict:
    start = time.perf_counter()

    X = embedder.encode([text])
    probs = clf.predict_proba(X)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Sort intents by probability descending
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

# ── Routes ───────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "User Intent Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "POST /classify":       "Classify a single message",
            "POST /classify/batch": "Classify multiple messages",
            "GET  /intents":        "List all supported intents",
            "GET  /health":         "Health check",
            "GET  /docs":           "Interactive API docs (Swagger)",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    return HealthResponse(
        status="ok",
        model_accuracy=metadata["accuracy"],
        num_intents=len(metadata["classes"]),
        num_training_samples=metadata["num_samples"],
    )


@app.get("/intents", tags=["Info"])
def list_intents():
    """Returns all supported intent categories with routing info."""
    return {
        "count": len(metadata["intents"]),
        "intents": metadata["intents"],
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
def classify(req: ClassifyRequest):
    """
    Classify a single user message into an intent.

    - **text**: The user's message (1–1000 chars)
    - **top_k**: How many top intents to return (default 3)

    Returns the predicted intent, confidence score, routing destination,
    and confidence tier (high / medium / low).
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty or whitespace.")
    return predict_single(req.text.strip(), top_k=req.top_k)


@app.post("/classify/batch", tags=["Classification"])
def classify_batch(req: BatchRequest):
    """
    Classify multiple messages at once (max 50).
    Returns a list of classification results.
    """
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty.")

    results = []
    for text in req.texts:
        if text.strip():
            results.append(predict_single(text.strip()).model_dump())

    return {
        "count": len(results),
        "results": results,
    }