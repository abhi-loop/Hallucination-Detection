# Hallucination Detection Using EigenScore

A full-stack implementation for detecting hallucinations in Large Language Models (LLMs) using the **EigenScore** metric — with a React dashboard for real-time analysis.

## 📖 Overview

Hallucinations in LLMs occur when models generate plausible-sounding but factually incorrect or inconsistent information. This project implements EigenScore, which quantifies hallucination likelihood by:

1. Generating K=10 responses to the same prompt (stochastic sampling)
2. Extracting hidden-state embeddings from each response (middle layer)
3. Computing eigenvalues of the embedding covariance matrix
4. Measuring variance through mean log eigenvalue (EigenScore)
5. Classifying against a threshold calibrated on TruthfulQA

**Lower EigenScore** → responses are consistent → likely **Factual**  
**Higher EigenScore** → responses are diverse → likely **Hallucination**

---

## 🏗️ Project Structure

```
hallucination-detection/
├── api/
│   └── server.py                  # FastAPI REST server (POST /analyze)
├── models/
│   ├── llm_loader.py              # Model loading with 4-bit quantization
│   ├── hidden_extraction.py       # Embedding extraction from model layers
│   └── generation.py              # Response generation with sampling
├── metrics/
│   ├── eigenscore.py              # EigenScore computation
│   ├── feature_clipping.py        # Feature Clipping (FC) — test-time embedding normalisation
│   ├── evaluation.py              # Evaluation utilities
│   └── threshold.py               # Optimal threshold via ROC curve
├── pipeline/
│   └── run_dataset.py             # Offline TruthfulQA pipeline → results.csv
├── data/
│   ├── loader.py                  # TruthfulQA dataset loader
│   ├── labeler.py                 # ROUGE-L + semantic similarity labeler
│   └── results.csv                # Calibration data (eigenscore + label per question)
├── hallucination-sentinel-main/   # React + Vite frontend dashboard
│   └── src/
│       ├── pages/Index.tsx        # Main page (chat + metrics)
│       ├── components/
│       │   ├── HallucinationVerdict.tsx   # Verdict card (eigenscore vs threshold)
│       │   ├── EigenScoreHistogram.tsx    # Reference distribution + live marker
│       │   ├── EmbeddingScatter.tsx       # Live PCA scatter of K embeddings
│       │   ├── ResponsesPanel.tsx         # Collapsible K-response list
│       │   └── ErrorBoundary.tsx          # Crash recovery
│       └── lib/
│           ├── api.ts             # Typed fetch wrapper for /analyze
│           └── mockData.ts        # Shared TypeScript types
├── main.py                        # CLI inference script
└── README.md
```

---

## 🔧 Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: Minimum 5 GB (4-bit quantized OPT-6.7B)
- **RAM**: ~20 GB (CPU offload enabled)
- **Storage**: ~13 GB for cached model weights

### Software
- Python 3.8+ with virtual environment
- Node.js 18+ and npm
- CUDA-compatible PyTorch

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/abhi-loop/Hallucination-Detection.git
cd Hallucination-Detection
```

### 2. Python Virtual Environment
```bash
# Windows
python -m venv env
env\Scripts\activate

# Linux/Mac
python3 -m venv env
source env/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
pip install sentence-transformers rouge-score scikit-learn datasets
pip install fastapi "uvicorn[standard]" pydantic
```

### 4. Install Frontend Dependencies
```bash
cd hallucination-sentinel-main
npm install
cd ..
```

---

## 🚀 Running the Full Stack

### Step 1 — Generate calibration data (first time only)
```bash
python pipeline/run_dataset.py --limit 100
```
This runs the EigenScore pipeline over 100 TruthfulQA questions and saves results to `data/results.csv`. Takes ~1–2 hours depending on GPU.

### Step 2 — Start the backend API server
```bash
# Terminal 1 (from project root, with env activated)
env\scripts\uvicorn api.server:app --port 8000
```
Wait for: **`[INFO] Model loaded. Server is ready.`** (~1–2 min)

### Step 3 — Start the frontend
```bash
# Terminal 2
cd hallucination-sentinel-main
npm run dev
```
Open **http://localhost:8080** in your browser.

### Step 4 — Analyze a question
Type any question in the chat panel and hit Send. After ~30–60 seconds:
- **Verdict card** — FACTUAL / HALLUCINATED with live eigenscore vs threshold
- **EigenScore Histogram** — reference distribution with live score marked
- **Embedding Scatter** — PCA of the 10 live response embeddings (PC1 vs PC2)
- **Generated Responses** — expandable list of all K=10 raw model answers
- **FEAT. CLIPPING toggle** (header bar) — enable/disable Feature Clipping on embeddings in real time (defaults to ON)

---

## 🖥️ CLI Usage (no frontend)

```bash
python main.py
```
Runs a single question, prints eigenscore, threshold, and verdict to terminal.

---

## 🧮 How EigenScore Works

```
Given K embeddings: {e₁, e₂, ..., eₖ} ∈ ℝᵈ

1. Center:      zᵢ = eᵢ - (1/K)∑eⱼ
2. Covariance:  Σ  = ZZᵀ + αI   (α = 1e-3 for numerical stability)
3. Eigenvalues: λ₁, ..., λₖ = eigvalsh(Σ)
4. Score:       s  = (1/K) ∑ log(max(λᵢ, ε))
```

### Threshold Calibration
The decision threshold is computed from `data/results.csv` using the **G-mean optimal point** on the ROC curve (`sklearn.metrics.roc_curve`). A live eigenscore above the threshold → hallucination; below → factual.

### Labeling (offline pipeline only)
Each TruthfulQA question is labelled by checking all K responses against the dataset's correct answers using:
1. **ROUGE-L ≥ 0.5** — longest-common-subsequence word overlap
2. **Semantic similarity ≥ 0.9** — cosine similarity via `nli-roberta-large`

If any one response passes either check → question labelled **Factual (0)**; otherwise **Hallucination (1)**.

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named uvicorn` | Use `env\scripts\uvicorn` directly (not `python -m uvicorn`) |
| `PackageNotFoundError: bitsandbytes` | Activate venv before running: `env\Scripts\activate` |
| Page goes blank after question | Backend crashed mid-inference; restart without `--reload` flag |
| `Cannot reach backend` | Ensure FastAPI server is running on port 8000 and model has finished loading |
| CUDA out of memory | Close other GPU processes; check `nvidia-smi` |
| Model download fails | Set `HF_TOKEN` env var or download manually from Hugging Face Hub |

---

## 📚 Technical Stack

| Layer | Technology |
|-------|-----------|
| LLM | Facebook OPT-6.7B (4-bit NF4 quantization via bitsandbytes) |
| Embeddings | Middle hidden layer of OPT-6.7B |
| Threshold | scikit-learn ROC curve (G-mean optimal) |
| Labeling | ROUGE-L + nli-roberta-large semantic similarity |
| API | FastAPI + uvicorn |
| Frontend | React 18 + Vite + TypeScript + Tailwind CSS |
| Charts | Recharts (SVG-based) |
| PCA | scikit-learn PCA(n_components=2) on live embeddings |

---

## 🙏 Acknowledgments

- EigenScore methodology from hallucination detection research
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for model infrastructure
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for 4-bit quantization
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa) for calibration dataset

---

**Note**: First run downloads ~13 GB of model weights. Subsequent runs load from cache instantly.