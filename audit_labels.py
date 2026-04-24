"""
audit_labels.py
===============
Manually audits the labels in tqaresults.csv by comparing
model responses against correct answers.

For FACTUAL (0): shows which response triggered the match and how.
For HALLUCINATION (1): shows all responses vs GT to check for missed matches.

Usage:
    python audit_labels.py
"""

import json
import csv
import re
import sys
import random
from rouge_score import rouge_scorer

# Force UTF-8 output on Windows (avoids cp1252 errors with special chars in GT aliases)
sys.stdout.reconfigure(encoding="utf-8")

DATA   = "data"
JSONL  = f"{DATA}/tqaraw_data.jsonl"
CSV    = f"{DATA}/tqaresults.csv"
ROUGE_THRESH = 0.2

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# ── Load data ────────────────────────────────────────────────────────────────
raw = {}
with open(JSONL, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line.strip())
        raw[r["question_id"]] = r

with open(CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# ── Helper ───────────────────────────────────────────────────────────────────
def norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()

def match_reason(resp, gts):
    r = norm(resp)
    for gt in gts:
        g = norm(gt)
        if g and g in r:
            return f"SUBSTRING:{gt}"
    for gt in gts:
        g = norm(gt)
        sc = scorer.score(g, r)["rougeL"].fmeasure
        if sc >= ROUGE_THRESH:
            return f"ROUGE={sc:.2f}:{gt}"
    return None

# ── Audit FACTUAL (0): check for false positives ─────────────────────────────
print("=" * 80)
print("LABELED FACTUAL (0) — verifying correct matches (any false positives?)")
print("=" * 80)
factual_rows = [r for r in rows if r["label"] == "0"]
print(f"Total factual: {len(factual_rows)}\n")

for i, row in enumerate(factual_rows):
    rec   = raw.get(row["question_id"], {})
    gts   = rec.get("correct_answers", [])
    resps = [r["text"] for r in rec.get("responses", [])]

    passing = []
    for resp in resps:
        reason = match_reason(resp, gts)
        if reason:
            passing.append((resp, reason))

    print(f"[F-{i+1:02d}] Q : {row['question'][:85]}")
    print(f"       GT: {gts[:4]}")
    if passing:
        resp, reason = passing[0]
        print(f"       MATCH ({reason}): {resp[:90]}")
    else:
        print(f"       !! NO MATCH FOUND — possible labeling error !!")
    print()

# ── Audit HALLUCINATION (1): check for false negatives ───────────────────────
print("=" * 80)
print("LABELED HALLUCINATION (1) — spot checking for missed matches (false negatives?)")
print("=" * 80)
hall_rows = [r for r in rows if r["label"] == "1"]
print(f"Total hallucination: {len(hall_rows)}  (showing 20 random samples)\n")

random.seed(42)
sample = random.sample(hall_rows, min(20, len(hall_rows)))

for i, row in enumerate(sample):
    rec   = raw.get(row["question_id"], {})
    gts   = rec.get("correct_answers", [])
    resps = [r["text"] for r in rec.get("responses", [])]

    print(f"[H-{i+1:02d}] Q : {row['question'][:85]}")
    print(f"       GT: {gts[:3]}")
    for j, resp in enumerate(resps[:4]):
        reason = match_reason(resp, gts)
        tag = f"MATCHED ({reason})" if reason else "no match"
        print(f"       R{j+1}: [{tag}] {resp[:80]}")
    print()
