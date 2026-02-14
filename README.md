# Hallucination Detection Using EigenScore

A Python-based implementation for detecting hallucinations in Large Language Models (LLMs) using the **EigenScore** metric. This project measures uncertainty and inconsistency in model outputs by analyzing the variance across multiple sampled responses.

## 📖 Overview

Hallucinations in LLMs occur when models generate plausible-sounding but factually incorrect or inconsistent information. This project implements EigenScore, which quantifies hallucination likelihood by:

1. Generating multiple responses to the same prompt
2. Extracting semantic embeddings from each response
3. Computing eigenvalues of the covariance matrix
4. Measuring variance through mean log eigenvalue

**Lower EigenScore** → Higher consistency (less hallucination)  
**Higher EigenScore** → Higher variance (potential hallucination)

## 🎯 Features

- **4-bit Quantization**: Efficiently loads large models (6.7B parameters) with ~75% memory reduction
- **Semantic Analysis**: Extracts middle-layer embeddings for robust representation
- **Numerical Stability**: Includes regularization and eigenvalue clamping to prevent computation errors
- **Configurable Sampling**: Adjustable temperature and sample count for response generation

## 🏗️ Project Structure

```
hallucination-detection/
├── models/
│   ├── llm_loader.py          # Model loading with 4-bit quantization
│   ├── hidden_extraction.py   # Embedding extraction from model layers
│   └── generation.py           # Response generation with sampling
├── metrics/
│   ├── eigenscore.py          # EigenScore computation
│   ├── evaluation.py          # Evaluation utilities
│   └── threshold.py           # Threshold detection
├── data/                      # Data storage directory
├── main.py                    # Main execution script
└── README.md                  # Project documentation
```

## 🔧 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: Minimum 4GB (for 4-bit quantized 6.7B model)
- **Storage**: ~13GB for cached model weights

### Software Requirements
- Python 3.8+
- CUDA-compatible PyTorch
- Git (for cloning the repository)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/abhi-loop/Hallucination-Detection.git
cd Hallucination-Detection
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv env
env\Scripts\activate

# Linux/Mac
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
```

**Key Dependencies:**
- `torch` - Deep learning framework
- `transformers` - Hugging Face transformers library
- `accelerate` - Model distribution and optimization
- `bitsandbytes` - 4-bit quantization support

## 🚀 Usage

### Basic Execution

Run the main script to generate responses and compute EigenScore:

```bash
python main.py
```

**Expected Output:**
```
Loading model...
Loading weights: 100%|████████| 516/516 [00:20<00:00]
Generating 5 different responses...
Extracting embeddings from each response...
  Response 1: What is the capital of France? Answer in one word. Paris...
  Response 2: What is the capital of France? Answer in one word. Paris...
  ...
Computing EigenScore...
Final EigenScore: -1.7792285680770874
```

### Customizing Parameters

#### Change the Question
Edit [`main.py`](main.py) line 12:
```python
question = "Your custom question here?"
```

#### Adjust Number of Samples
Edit [`main.py`](main.py) line 16:
```python
K = 10  # Generate 10 responses instead of 5
```

#### Modify Temperature (Response Randomness)
Edit [`models/generation.py`](models/generation.py) line 15:
```python
temperature=0.1  # Low (deterministic) → High (creative)
```

**Temperature Guide:**
- `0.1-0.3`: Very consistent responses (low EigenScore)
- `0.5-0.7`: Balanced creativity and consistency
- `0.9-1.5`: Highly varied responses (high EigenScore)

## 🧮 How EigenScore Works

### Algorithm Steps

1. **Response Generation**: Sample K responses using temperature-based sampling
2. **Embedding Extraction**: Extract semantic embeddings from model's middle layer
3. **Centering**: Subtract mean across samples: `Z = embeddings - mean(embeddings)`
4. **Covariance**: Compute `Σ = Z @ Z^T` (K×K matrix)
5. **Regularization**: Add `α·I` for numerical stability
6. **Eigenvalue Decomposition**: Compute eigenvalues of Σ
7. **Score**: `EigenScore = mean(log(eigenvalues))`

### Mathematical Formulation

```
Given K embeddings: {e₁, e₂, ..., eₖ} ∈ ℝᵈ

1. Center: zᵢ = eᵢ - (1/K)∑eⱼ
2. Covariance: Σ = ZZᵀ + αI  (where Z = [z₁, ..., zₖ])
3. Eigenvalues: λ₁, λ₂, ..., λₖ = eig(Σ)
4. Score: s = (1/K)∑log(λᵢ)
```

## 🔍 Example Results

### High Consistency (Low EigenScore)
```
Temperature: 0.1
Responses: All answer "Paris"
EigenScore: -1.78
```

### Moderate Variance
```
Temperature: 0.7
Responses: Mix of "Paris", elaborations, questions
EigenScore: 3.94
```

### High Variance (Potential Hallucination)
```
Temperature: 1.5
Responses: Diverse, sometimes incorrect
EigenScore: 7.2+ (hypothetical)
```

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce model size or switch to smaller model
- Ensure no other GPU processes are running
- Try `nvidia-smi` to check VRAM usage

**2. `bitsandbytes` Import Error**
- Windows users: Install from source or use WSL
- Ensure CUDA toolkit is properly installed

**3. Model Download Fails**
- Check internet connection
- Set Hugging Face token: `export HF_TOKEN=your_token`
- Try manual download from [Hugging Face Hub](https://huggingface.co/facebook/opt-6.7b)

**4. NaN EigenScore**
- Already fixed in current version with eigenvalue clamping
- If persists, increase `alpha` in `eigenscore.py`

## 📚 Technical Details

### Model Architecture
- **Default Model**: Facebook OPT-6.7B
- **Quantization**: 4-bit NormalFloat (NF4) with double quantization
- **Embedding Layer**: Middle layer (layer 16/32) for semantic richness
- **Device Mapping**: Automatic distribution across available GPUs

### Numerical Stability Enhancements
- **Regularization**: `α = 0.01` prevents singular covariance
- **Eigenvalue Clamping**: `min = 1e-6` prevents log(0) or log(negative)
- **FP16 Compute**: Mixed precision for efficiency

## 🙏 Acknowledgments

- EigenScore methodology from hallucination detection research
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for model infrastructure
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization support



---

**Note**: First run downloads ~13GB model weights. Subsequent runs load from cache instantly.