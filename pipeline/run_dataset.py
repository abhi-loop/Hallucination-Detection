"""
pipeline/run_dataset.py

TriviaQA Hallucination Detection Pipeline
==========================================
Runs the full pipeline over the TriviaQA (rc.nocontext) dataset.

For each question:
  1. Generates K = 10 diverse responses + per-token log-probs
  2. Computes: EigenScore, LN-Entropy, avg_token_prob, Lexical Similarity,
               length_mean, length_std
  3. Labels (0 = factual, 1 = hallucination) via ROUGE-L + semantic similarity
  4. Saves incrementally to:
       data/tqaraw_data.jsonl  — one JSON line per question (question + responses + logprobs)
       data/tqaresults.csv     — one row per question (all metrics)

Resume: already-processed question_ids are read from tqaraw_data.jsonl at startup.
        Questions already in the file are skipped without re-generating.

Usage:
    python pipeline/run_dataset.py --limit 50
    python pipeline/run_dataset.py           # process entire TriviaQA validation set
"""

import os
import sys
import csv
import json
import argparse
import statistics

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_loader import load_model
from models.generation import generate_k_answers_with_logprobs
from metrics.eigenscore import compute_eigenscore
from metrics.feature_clipping import FeatureClipping
from metrics.ln_entropy import compute_ln_entropy
from metrics.avg_token_prob import compute_avg_token_prob
from metrics.lexical_similarity import compute_lexical_similarity
from data.loader import load_tqa
from data.labeler import label_question, response_is_correct

# ── Output paths ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(_ROOT, "data")
JSONL_PATH  = os.path.join(DATA_DIR, "tqaraw_data.jsonl")
CSV_PATH    = os.path.join(DATA_DIR, "tqaresults.csv")

K = 10  # responses per question

CSV_FIELDNAMES = [
    "question_index",
    "question_id",
    "question",
    "eigenscore",
    "ln_entropy",
    "avg_token_prob",
    "lexical_similarity",
    "length_mean",
    "length_std",
    "label",
]


# ── Resume helpers ────────────────────────────────────────────────────────────

def load_processed_ids() -> set:
    """
    Read raw_data.jsonl and return the set of question_ids already processed.
    Safe to call if the file does not yet exist.
    """
    processed = set()
    if not os.path.exists(JSONL_PATH):
        return processed
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                processed.add(obj["question_id"])
            except (json.JSONDecodeError, KeyError):
                pass  # skip malformed lines
    return processed


def append_jsonl(record: dict):
    """Append one JSON record as a single line to raw_data.jsonl."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(row: dict):
    """Append one metrics row to nqresults.csv (writes header if new file)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_length_stats(responses: list[dict]) -> tuple[float, float]:
    """
    Character-level length mean and std across K response texts.
    Returns (mean, std). std is 0.0 if only one response.
    """
    lengths = [len(r["text"]) for r in responses if r.get("text")]
    if not lengths:
        return 0.0, 0.0
    mean = statistics.mean(lengths)
    std  = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    return mean, std


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(limit: int | None):
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Load previously processed IDs for resume ──────────────────────────
    processed_ids = load_processed_ids()
    if processed_ids:
        print(f"[Resume] {len(processed_ids)} questions already processed — will skip them.")

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    tokenizer, model = load_model()

    # Feature clipping — shared across all questions (memory bank accumulates)
    clipper = FeatureClipping(memory_size=3000, percentile=0.2)
    print("[INFO] Feature Clipping enabled (memory_size=3000, percentile=0.2%)")

    # ── Load NQ ───────────────────────────────────────────────────────────
    print(f"Loading TriviaQA (limit={limit})...")
    records = load_tqa(limit=limit)
    print(f"Loaded {len(records)} TriviaQA questions.\n")

    # ── Process each question ─────────────────────────────────────────────
    global_index = 0  # tracks absolute position in the NQ slice

    for record in records:
        question_id   = record["question_id"]
        question      = record["question"]
        correct_answers = record["correct_answers"]

        if question_id in processed_ids:
            print(f"[{global_index}] Skipping (already done): {question[:70]}...")
            global_index += 1
            continue

        print(f"[{global_index}] {question[:90]}")

        # ── 1. Generate K responses + log-probs + hidden-state embeddings ──
        # Embeddings are captured during generation (middle hidden layer,
        # last-token pooling) — NOT re-encoded from text.
        responses, embeddings, gen_config = generate_k_answers_with_logprobs(
            model, tokenizer, question, k=K, clipper=clipper
        )

        # ── 2. Extract texts for convenience ──────────────────────────────
        texts = [r["text"] for r in responses]

        # ── 3. Compute metrics ────────────────────────────────────────────

        # EigenScore — uses hidden-state embeddings from the generation pass
        eigenscore = compute_eigenscore(embeddings)

        # Uncertainty / confidence metrics (from log-probs)
        ln_entropy      = compute_ln_entropy(responses)
        avg_tok_prob    = compute_avg_token_prob(responses)

        # Lexical consistency across responses
        lex_sim         = compute_lexical_similarity(responses)

        # Response length stats
        length_mean, length_std = compute_length_stats(responses)

        # ── 4. Label ──────────────────────────────────────────────────────
        label = label_question(texts, correct_answers)

        # ── 5. Logging ────────────────────────────────────────────────────
        print(f"   Ground truth   : {correct_answers[:3]}")
        for j, (resp, txt) in enumerate(zip(responses, texts)):
            per_label   = 0 if response_is_correct(txt, correct_answers) else 1
            verdict_per = "Factual" if per_label == 0 else "Hallucination"
            n_lp        = len(resp["token_logprobs"])
            print(f"   Response [{j+1:02d}]: {txt[:100]}  →  {verdict_per}  (tokens={n_lp})")

        verdict = "Factual" if label == 0 else "Hallucination"
        print(
            f"   EigenScore={eigenscore:.4f} | LN-Entropy={ln_entropy:.4f} | "
            f"avg_token_prob={avg_tok_prob:.4f} | Lex-Sim={lex_sim:.4f} | "
            f"len_mean={length_mean:.1f} | len_std={length_std:.1f} | "
            f"label={label} ({verdict})"
        )

        # ── 6. Save raw data (JSONL) ──────────────────────────────────────
        raw_record = {
            "question_index":  global_index,
            "question_id":     question_id,
            "question":        question,
            "correct_answers": correct_answers,
            "generation_config": gen_config,
            "responses":       responses,   # list of {text, token_logprobs}
        }
        append_jsonl(raw_record)

        # ── 7. Save metrics (CSV) ─────────────────────────────────────────
        csv_row = {
            "question_index":   global_index,
            "question_id":      question_id,
            "question":         question,
            "eigenscore":       round(eigenscore,    6),
            "ln_entropy":       round(ln_entropy,    6),
            "avg_token_prob":   round(avg_tok_prob,  6),
            "lexical_similarity": round(lex_sim,     6),
            "length_mean":      round(length_mean,   2),
            "length_std":       round(length_std,    2),
            "label":            label,
        }
        append_csv(csv_row)

        processed_ids.add(question_id)
        global_index += 1
        print()

    print(f"Done.")
    print(f"  Raw data  → {JSONL_PATH}")
    print(f"  Metrics   → {CSV_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NQ hallucination detection pipeline"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of NQ questions to process (default: all answerable questions)",
    )
    args = parser.parse_args()
    run(limit=args.limit)
