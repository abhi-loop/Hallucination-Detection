"""
api/server.py

FastAPI server that exposes POST /analyze — runs the real EigenScore inference
pipeline and returns results the React frontend can consume.

Start with:
    python -m uvicorn api.server:app --reload --port 8000
"""

import os
import csv
import sys
from contextlib import asynccontextmanager

# Allow imports from project root when run as `python -m uvicorn api.server:app`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models.llm_loader import load_model
from models.generation import generate_k_answers_with_logprobs, generate_canonical_answer
from metrics.eigenscore import compute_eigenscore
from metrics.threshold import find_best_threshold
from metrics.feature_clipping import FeatureClipping

# ── Path to calibration data ──────────────────────────────────────────────────
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "tfulresults.csv"
)

# ── Fallback calibration data used when results.csv is missing / too small ────
_FALLBACK_SCORES = [-1.8, -1.5, -1.6, -1.7, -1.4, 2.0, 2.3, 2.8]
_FALLBACK_LABELS = [0, 0, 0, 0, 0, 1, 1, 1]


# ── Global model state (loaded once at startup) ────────────────────────────────
_state: dict = {}


def load_reference_scores():
    """Load EigenScores + labels from data/results.csv, or fall back to stubs."""
    if not os.path.exists(RESULTS_PATH):
        print("[WARNING] data/results.csv not found. Using placeholder reference scores.")
        return _FALLBACK_SCORES, _FALLBACK_LABELS

    scores, labels = [], []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["eigenscore"]))
            labels.append(int(row["label"]))

    if len(scores) < 2:
        print("[WARNING] results.csv has fewer than 2 rows. Using placeholder reference scores.")
        return _FALLBACK_SCORES, _FALLBACK_LABELS

    print(f"[INFO] Loaded {len(scores)} reference scores from data/results.csv")
    return scores, labels


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release on shutdown."""
    print("[INFO] Loading model — this can take 1-2 minutes ...")
    tokenizer, model = load_model()
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    # One FC instance per server session — memory bank grows across queries
    _state["clipper"] = FeatureClipping(memory_size=3000, percentile=0.2)
    print("[INFO] Model loaded. Feature Clipping enabled. Server is ready.")
    yield
    _state.clear()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Hallucination Sentinel API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allows ngrok / Colab tunnel URLs in addition to localhost
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to analyze")
    k: int = Field(10, ge=1, le=20, description="Number of responses to generate")
    use_clipping: bool = Field(True, description="Whether to apply Feature Clipping to embeddings")


class Embedding2D(BaseModel):
    x: float
    y: float
    label: str
    cluster: int


class AnalyzeResponse(BaseModel):
    eigenscore: float
    threshold: float
    g_mean: float
    verdict: str              # "factual" | "hallucination"
    confidence: float         # 0-1, absolute distance from threshold mapped to [0,1]
    canonical_response: str   # single greedy (do_sample=False) answer shown in chat
    responses: list[str]      # all K sampled answers (used for EigenScore)
    eigen_scores_ref: list[float]     # reference scores from results.csv (for histogram)
    embeddings_2d: list[Embedding2D]  # 2-D scatter data


# ── Helper: real PCA on the K live embedding tensors ─────────────────────────

def _pca_embeddings_2d(embeddings: list) -> list[Embedding2D]:
    """
    Runs sklearn PCA (n_components=2) on the K live response embeddings
    and returns 2D coordinates for the scatter chart.
    Each point is labelled "Response N" so the tooltip is informative.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    # Stack to (K, d) float32 numpy array on CPU
    mat = np.stack([e.cpu().float().numpy() for e in embeddings])  # (K, d)

    # Need at least 2 unique embeddings to do PCA
    n_components = min(2, mat.shape[0] - 1)
    if n_components < 1:
        # Degenerate case: only 1 sample or all identical
        return [
            Embedding2D(x=0.0, y=0.0, label=f"Response {i+1}", cluster=0)
            for i in range(mat.shape[0])
        ]

    coords = PCA(n_components=n_components).fit_transform(mat)  # (K, 2)

    return [
        Embedding2D(
            x=float(coords[i, 0]),
            y=float(coords[i, 1]) if coords.shape[1] > 1 else 0.0,
            label=f"Response {i+1}",
            cluster=0,   # single cluster — coloured uniformly
        )
        for i in range(len(embeddings))
    ]


# ── Endpoint ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again.")

    tokenizer = _state["tokenizer"]
    model = _state["model"]
    clipper = _state["clipper"]

    # 1a. Generate one greedy (deterministic) response for display in chat
    canonical_response = generate_canonical_answer(model, tokenizer, req.question)

    # 1b. Generate K sampled responses + embeddings from the generation pass
    active_clipper = clipper if req.use_clipping else None
    responses, embeddings, _ = generate_k_answers_with_logprobs(
        model, tokenizer, req.question, k=req.k, clipper=active_clipper
    )

    # Extract text strings for serialization
    response_texts = [r["text"] for r in responses]

    # 3. Compute EigenScore for the live question
    eigenscore = compute_eigenscore(embeddings)

    # 4. Calibrate threshold from results.csv
    ref_scores, ref_labels = load_reference_scores()
    threshold, g_mean = find_best_threshold(ref_scores, ref_labels)

    # 5. Classify
    is_hallucination = eigenscore > threshold
    verdict = "hallucination" if is_hallucination else "factual"

    # 6. Confidence: map |score - threshold| to a 0-1 scale (capped at 1)
    score_range = max(abs(max(ref_scores) - threshold), abs(min(ref_scores) - threshold), 1e-6)
    confidence = min(abs(eigenscore - threshold) / score_range, 1.0)

    # 7. Run PCA on the live embeddings → 2D scatter
    embeddings_2d = _pca_embeddings_2d(embeddings)

    return AnalyzeResponse(
        eigenscore=round(eigenscore, 6),
        threshold=round(float(threshold), 6),
        g_mean=round(float(g_mean), 6),
        verdict=verdict,
        confidence=round(confidence, 4),
        canonical_response=canonical_response,
        responses=response_texts,
        eigen_scores_ref=ref_scores,
        embeddings_2d=embeddings_2d,
    )
