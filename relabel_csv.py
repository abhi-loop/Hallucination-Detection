"""
relabel_csv.py
==============
Re-labels every row in data/tqaresults.csv using fast substring + ROUGE-L only.
Skips the semantic similarity model entirely — runs in seconds, not minutes.

Reads:
    data/tqaraw_data.jsonl  — source of truth (question_id, correct_answers, responses)
Writes:
    data/tqaresults.csv     — label column updated in-place
    data/tqaresults.csv.bak — backup of the original file

Usage:
    python relabel_csv.py
"""

import os
import re
import csv
import json
import shutil
from collections import Counter
from rouge_score import rouge_scorer

_ROOT    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_ROOT, "data")
JSONL    = os.path.join(DATA_DIR, "tqaraw_data.jsonl")
CSV      = os.path.join(DATA_DIR, "tqaresults.csv")
BAK      = CSV + ".bak"

# Same ROUGE-L threshold as the updated labeler
ROUGE_L_THRESHOLD = 0.2


# ── Fast labeling (no semantic model) ────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _response_correct_fast(response: str, correct_answers: list) -> bool:
    """Substring check then ROUGE-L — no neural model, runs in microseconds."""
    resp = _normalize(response)
    if not resp:
        return False

    # Strategy 1: verbatim substring
    for gt in correct_answers:
        gt_n = _normalize(gt)
        if gt_n and gt_n in resp:
            return True

    # Strategy 2: ROUGE-L F1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for gt in correct_answers:
        gt_n = _normalize(gt)
        if scorer.score(gt_n, resp)["rougeL"].fmeasure >= ROUGE_L_THRESHOLD:
            return True

    return False


def label_fast(responses: list, correct_answers: list) -> int:
    """Returns 0 (factual) if any response passes, else 1 (hallucination)."""
    for r in responses:
        if _response_correct_fast(r, correct_answers):
            return 0
    return 1


# ── 1. Back up the original CSV ──────────────────────────────────────────────
shutil.copy2(CSV, BAK)
print(f"[backup] {CSV} -> {BAK}")


# ── 2. Load raw data keyed by question_id ────────────────────────────────────
print(f"[load]   Reading {JSONL} ...")
raw = {}
with open(JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            qid = rec["question_id"]
            raw[qid] = {
                "correct_answers": rec["correct_answers"],
                "responses":       [r["text"] for r in rec.get("responses", [])],
            }
        except (json.JSONDecodeError, KeyError):
            pass

print(f"[load]   {len(raw)} raw records loaded.")


# ── 3. Re-label ──────────────────────────────────────────────────────────────
print(f"[label]  Re-labeling rows in {CSV} ...")

with open(CSV, "r", encoding="utf-8", newline="") as f:
    reader     = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows       = list(reader)

old_counts = Counter(int(r["label"]) for r in rows)
changed    = 0

for i, row in enumerate(rows):
    qid = row["question_id"]
    if qid not in raw:
        print(f"  [WARN] question_id={qid} not found in JSONL — skipping.")
        continue

    entry     = raw[qid]
    old_label = int(row["label"])
    new_label = label_fast(entry["responses"], entry["correct_answers"])

    if new_label != old_label:
        direction = "hall->fact" if old_label == 1 else "fact->hall"
        print(f"  [{i:>3}] {direction} | {row['question'][:70]}")
        changed += 1

    row["label"] = new_label

new_counts = Counter(int(r["label"]) for r in rows)


# ── 4. Write updated CSV ─────────────────────────────────────────────────────
with open(CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"Done. {changed} / {len(rows)} labels changed.")
print()
print(f"  Before:  Factual={old_counts[0]}  Hallucination={old_counts[1]}")
print(f"  After:   Factual={new_counts[0]}  Hallucination={new_counts[1]}")
print()
print(f"  Updated CSV -> {CSV}")
print(f"  Backup      -> {BAK}")
