# Hallucination Sentinel

> **EigenScore-based LLM hallucination detection** — full-stack system with a React dashboard, FastAPI inference server, Feature Clipping, and multiple deployment options.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![React 18](https://img.shields.io/badge/react-18-61DAFB)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

Hallucinations in LLMs occur when models generate plausible-sounding but factually incorrect or inconsistent information. This project detects them using **EigenScore**, a training-free metric that exploits the geometric structure of hidden-state embeddings:

1. Generate **K = 10** responses to the same prompt (stochastic sampling)
2. Extract hidden-state embeddings from the **middle layer** of OPT-6.7B
3. Compute the **covariance matrix** of the centred embeddings
4. Take the **mean log eigenvalue** → EigenScore
5. Classify against a **G-mean optimal threshold** calibrated on TruthfulQA

**Lower EigenScore** → responses are tightly clustered → likely **Factual**  
**Higher EigenScore** → responses are diverse / scattered → likely **Hallucination**

**Feature Clipping (FC)** can be toggled at runtime to normalise embeddings before scoring, which sharpens the separation between factual and hallucinated clusters.

---

## 🏗️ Project Structure

```
hallucination-detection/
├── api/
│   └── server.py                  # FastAPI REST server — POST /analyze, GET /health
├── models/
│   ├── llm_loader.py              # OPT-6.7B loading with 4-bit NF4 quantisation
│   ├── hidden_extraction.py       # Hidden-state extraction from model middle layer
│   └── generation.py              # K-response sampling + embedding extraction
├── metrics/
│   ├── eigenscore.py              # EigenScore computation (covariance → log eigenvalues)
│   ├── feature_clipping.py        # Feature Clipping — online percentile-based normalisation
│   ├── avg_token_prob.py          # Average token log-probability baseline metric
│   ├── lexical_similarity.py      # Self-BLEU / pairwise lexical similarity
│   ├── ln_entropy.py              # Length-normalised token-level entropy
│   ├── evaluation.py              # AUROC / accuracy evaluation utilities
│   └── threshold.py               # Optimal threshold via ROC G-mean
├── pipeline/
│   └── run_dataset.py             # Offline TruthfulQA pipeline → data/tfulresults.csv
├── data/
│   ├── loader.py                  # TruthfulQA dataset loader (HuggingFace Datasets)
│   ├── labeler.py                 # ROUGE-L + semantic similarity auto-labeler
│   └── tfulresults.csv            # Calibration data (eigenscore + label per question)
├── hallucination-sentinel-main/   # React + Vite + TypeScript frontend
│   └── src/
│       ├── pages/Index.tsx        # Main page — chat panel + metrics dashboard
│       ├── components/
│       │   ├── HallucinationVerdict.tsx   # Verdict card (eigenscore vs threshold)
│       │   ├── EigenScoreHistogram.tsx    # Reference distribution + live score marker
│       │   ├── EmbeddingScatter.tsx       # Live PCA scatter of K embeddings
│       │   ├── ResponsesPanel.tsx         # Collapsible K-response list
│       │   └── ErrorBoundary.tsx          # Crash recovery wrapper
│       └── lib/
│           ├── api.ts             # Typed fetch wrapper (supports localStorage URL override)
│           └── mockData.ts        # Shared TypeScript types
├── hallucination_sentinel_colab.ipynb  # Google Colab deployment notebook (ngrok)
├── evaluate_tqa.py                # Standalone AUROC evaluation script
├── audit_labels.py                # Label audit / debug helper
├── relabel_csv.py                 # Re-labelling utility for results CSV
├── main.py                        # CLI inference script (no frontend needed)
└── requirements.txt               # Python dependencies with hardware notes
```

---

## 🔧 Prerequisites

### Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 5 GB | 8 GB+ |
| System RAM | 24 GB | 32 GB+ |
| Disk | 15 GB free | 20 GB+ |
| CUDA | 11.8 | 12.1 / 12.4 |

> **Note:** CPU-only inference is not supported. 4-bit quantisation requires CUDA.

### Software

- **Python** 3.10 – 3.12 (3.10 recommended)
- **Node.js** 18+ and npm 9+ (or bun ≥ 1.0)
- **NVIDIA driver** ≥ 520 (for CUDA 11.8) or ≥ 525 (for CUDA 12.x)

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

# Linux / macOS (macOS not supported for inference, only for dev)
python3 -m venv env
source env/bin/activate
```

### 3. Install Python Dependencies

```bash
# Install PyTorch with CUDA first (pick the wheel matching your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or cu121 / cu124 — check with: nvidia-smi

pip install -r requirements.txt
```

> **Windows note:** `bitsandbytes` requires Linux for full CUDA support.  
> Use `pip install bitsandbytes-windows` or run inside WSL2.

### 4. Install Frontend Dependencies

```bash
cd hallucination-sentinel-main
npm install
cd ..
```

---

## 🚀 Running the Full Stack (Local)

### Step 1 — Generate calibration data *(first time only)*

```bash
python pipeline/run_dataset.py --limit 817
```

Runs the EigenScore pipeline over 817 TruthfulQA questions and saves results to `data/tfulresults.csv`. Takes **~3-4 hours** depending on GPU.

### Step 2 — Start the backend API server

```bash
# Terminal 1 — from project root with env activated
env\scripts\uvicorn api.server:app --port 8000
```

Wait for: **`[INFO] Model loaded. Feature Clipping enabled. Server is ready.`** (~1–2 min)

### Step 3 — Start the React frontend

```bash
# Terminal 2
cd hallucination-sentinel-main
npm run dev
```

Open **http://localhost:8080** in your browser.

### Step 4 — Analyse a question

Type any question in the chat panel and press **Send**. After ~30–60 seconds:

| Panel | Description |
|-------|-------------|
| **Verdict card** | FACTUAL / HALLUCINATED with live eigenscore vs threshold |
| **EigenScore Histogram** | Reference distribution (TruthfulQA) with live score marked |
| **Embedding Scatter** | PCA of the 10 live response embeddings (PC1 vs PC2) |
| **Generated Responses** | Expandable list of all K = 10 raw model answers |
| **FC toggle** (header) | Enable/disable Feature Clipping on embeddings (default: ON) |

---

## 🌐 API Reference

The FastAPI server exposes two endpoints:

### `GET /health`

Returns server and model status.

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /analyze`

**Request body:**

```json
{
  "question": "Who wrote Hamlet?",
  "k": 10,
  "use_clipping": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | — | The question to analyse |
| `k` | int | 10 | Number of sampled responses (1–20) |
| `use_clipping` | bool | true | Apply Feature Clipping to embeddings |

## ☁️ Deployment

The frontend is deployed as a static site on **Vercel**. The GPU backend runs on **Google Colab** (T4 GPU) and is exposed to the internet via an **ngrok HTTPS tunnel**, which the Vercel frontend calls directly.

```
[Vercel — React frontend]  ──HTTPS──▶  [ngrok tunnel]  ──▶  [FastAPI on Colab T4 GPU]
```

### Step 1 — Create a lightweight project ZIP

Exclude large artefacts before uploading to Colab:

```powershell
# PowerShell — from project root
$exclude = @('env', 'node_modules', '.git', '__pycache__', '*.zip', 'unused_data')
Compress-Archive -Path (Get-ChildItem -Exclude $exclude) -DestinationPath hallucination-detection.zip
```

### Step 2 — Launch the backend on Google Colab

1. Open `hallucination_sentinel_colab.ipynb` in Colab.
2. Set the runtime to **GPU** (T4 or better): *Runtime → Change runtime type → T4 GPU*.
3. Run all cells in order:

   | Cell | What it does |
   |------|--------------|
   | **Step 1** | Mounts Google Drive; copies cached model weights to local disk for faster loading |
   | **Step 2A** | Upload `hallucination-detection.zip` *(first time only)* |
   | **Step 2B** | Load project from Drive *(subsequent sessions — skip 2A)* |
   | **Step 3** | Verify GPU + install all Python dependencies |
   | **Step 4** | Patch CORS to allow all origins (`*`) for ngrok |
   | **Step 5** | Paste your [ngrok auth token](https://dashboard.ngrok.com/get-started/your-authtoken) |
   | **Step 6** | Launch FastAPI server + create ngrok tunnel → prints the **PUBLIC URL** |
   | **Step 7** | Smoke test (`/health` + sample `/analyze` call) |

4. Once Step 6 completes you will see:
   ```
   ============================================================
     PUBLIC URL  : https://xxxx-xx-xx-xxx-xx.ngrok-free.app
     /health     : https://xxxx-xx-xx-xxx-xx.ngrok-free.app/health
     POST/analyze: https://xxxx-xx-xx-xxx-xx.ngrok-free.app/analyze
   ============================================================
   ```

> **Model caching:** weights (~13 GB) are downloaded once and persisted to `MyDrive/hf_cache`.  
> First run takes **15–30 min**; subsequent sessions copy from Drive in **5–8 min**.

### Step 3 — Deploy the frontend to Vercel

```bash
cd hallucination-sentinel-main
npx vercel deploy --prod
```

In the **Vercel dashboard → Project → Settings → Environment Variables**, add:

```
VITE_API_URL = https://xxxx-xx-xx-xxx-xx.ngrok-free.app
```

Then trigger a redeploy so the build picks up the variable. The frontend will call the Colab backend via the ngrok URL on every inference request.

> **Note:** The ngrok URL changes each time Step 6 is re-run (new Colab session). Update `VITE_API_URL` in Vercel and redeploy, **or** use the **Backend URL** field in the dashboard UI header — it overrides the env variable at runtime via `localStorage` without needing a redeploy.

### How the frontend resolves the backend URL

The frontend (`src/lib/api.ts`) picks the backend URL in this priority order:

1. `localStorage` key `backendUrl` — set via the "Backend URL" field in the UI header
2. `VITE_API_URL` env variable — baked in at Vercel build time
3. Hardcoded fallback `http://localhost:8000`

---

## 🧮 How EigenScore Works

```
Given K embeddings: {e₁, e₂, ..., eₖ} ∈ ℝᵈ

1. Center:      zᵢ = eᵢ − (1/K) ∑ eⱼ
2. Covariance:  Σ  = ZZᵀ + αI     (α = 1e-3 for numerical stability)
3. Eigenvalues: λ₁, ..., λₖ = eigvalsh(Σ)
4. Score:       s  = (1/K) ∑ log(max(λᵢ, ε))
```

### Threshold Calibration

The decision threshold is computed from `data/tfulresults.csv` using the **G-mean optimal point** on the ROC curve (`sklearn.metrics.roc_curve`). A live eigenscore **above** the threshold → hallucination; **below** → factual.

### Labeling (offline pipeline)

Each TruthfulQA question is labelled by checking all K responses against the dataset's correct answers:

1. **ROUGE-L ≥ 0.5** — longest-common-subsequence word overlap
2. **Semantic similarity ≥ 0.9** — cosine similarity via `nli-roberta-large`

If **any one** response passes either check → labelled **Factual (0)**; otherwise **Hallucination (1)**.

### Feature Clipping

FC maintains an online memory bank of embeddings across queries. At test time it clips each dimension to the [p, 1−p] percentile range observed in the bank (default p = 0.2). This removes outlier dimensions that inflate the covariance matrix and sharpen the factual / hallucinated cluster separation.

---

## 📚 Technical Stack

| Layer | Technology |
|-------|-----------|
| LLM | Facebook OPT-6.7B (4-bit NF4 quantisation via bitsandbytes) |
| Embeddings | Middle hidden layer of OPT-6.7B |
| Feature Clipping | Online percentile-based normalisation (custom implementation) |
| Threshold | scikit-learn ROC curve (G-mean optimal) |
| Labelling | ROUGE-L + `nli-roberta-large` semantic similarity |
| API | FastAPI + uvicorn (ASGI) |
| Frontend | React 18 + Vite 5 + TypeScript + Tailwind CSS + shadcn/ui |
| Charts | Recharts (SVG-based) |
| PCA | scikit-learn `PCA(n_components=2)` on live embeddings |
| Colab tunnel | pyngrok (ngrok v3) |

---

## 🙏 Acknowledgments

- EigenScore methodology from hallucination detection research
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for model infrastructure
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for 4-bit quantisation
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa) for calibration dataset
- [shadcn/ui](https://ui.shadcn.com/) for the React component library

---

> **First run:** downloads ~13 GB of model weights from HuggingFace. Subsequent runs load from cache instantly (~1–2 min).
