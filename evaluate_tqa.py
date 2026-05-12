"""
evaluate_tqa.py

Evaluates all metrics in data/tfulresults.csv individually via AUROC and
best-threshold G-mean, then reports a combined summary table.

Usage (from project root):
    python evaluate_tqa.py
"""

import sys
import pandas as pd

sys.path.insert(0, ".")

from metrics.threshold import find_best_threshold
from metrics.evaluation import compute_auroc

# ── Load results ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/tfulresults.csv").dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print(f"Loaded {len(df)} rows  |  Factual: {(df['label']==0).sum()}  |  Hallucination: {(df['label']==1).sum()}\n")

# ── Metrics to evaluate ───────────────────────────────────────────────────────
# For each metric, specify whether a HIGHER score predicts hallucination (True)
# or LOWER score predicts hallucination (False).
# roc_auc_score assumes higher score = more likely positive (hallucination=1).
# We flip the sign for metrics where lower = more uncertain.
METRICS = {
    "eigenscore":        {"flip": False},   # higher eigenscore → more uncertain → hallucination
    "ln_entropy":        {"flip": False},   # higher entropy    → more uncertain → hallucination
    "avg_token_prob":    {"flip": True},    # lower avg_prob    → more uncertain → hallucination (flip)
    "lexical_similarity":{"flip": True},    # lower lex-sim     → more diverse   → hallucination (flip)
    "length_mean":       {"flip": False},   # no strong prior; include raw
    "length_std":        {"flip": False},   # higher std        → more inconsistent
}

# ── Per-metric evaluation ─────────────────────────────────────────────────────
print(f"{'Metric':<22} {'AUROC':>7}  {'Threshold':>10}  {'G-mean':>7}")
print("-" * 55)

results = {}
for metric, cfg in METRICS.items():
    if metric not in df.columns:
        print(f"{metric:<22}  [column not found — skipping]")
        continue

    col = df[metric].dropna()
    valid_df = df.loc[col.index]
    scores = col.tolist()
    labels = valid_df["label"].tolist()

    if len(set(labels)) < 2:
        print(f"{metric:<22}  [only one class present — skipping]")
        continue

    # Flip sign so that higher always means "more likely hallucination"
    eval_scores = [-s for s in scores] if cfg["flip"] else scores

    auroc = compute_auroc(eval_scores, labels)
    threshold, gmean = find_best_threshold(eval_scores, labels)

    # Un-flip threshold for display
    display_threshold = -threshold if cfg["flip"] else threshold

    results[metric] = {"auroc": auroc, "threshold": display_threshold, "gmean": gmean}
    print(f"{metric:<22}  {auroc:>6.4f}   {display_threshold:>10.4f}   {gmean:>6.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
if results:
    print()
    best_metric = max(results, key=lambda m: results[m]["auroc"])
    print(f"Best single metric by AUROC : {best_metric}  ({results[best_metric]['auroc']:.4f})")
