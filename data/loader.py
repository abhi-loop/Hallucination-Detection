"""
data/loader.py

Loads datasets from HuggingFace for hallucination detection experiments.

load_tqa        — TriviaQA (rc.nocontext)
load_truthfulqa — TruthfulQA (generation config)
"""

from datasets import load_dataset


def load_tqa(split="validation", limit=None):
    """
    Load TriviaQA (rc.nocontext) from HuggingFace using streaming mode.
    Returns a list of dicts:
        {
            "question_id":    str,
            "question":       str,
            "correct_answers": [str, ...]   # normalized answer aliases
        }

    Args:
        split: HuggingFace dataset split to load ("train" or "validation").
               The validation split (~9.9 k questions) is the standard dev set.
        limit: if set, return at most this many questions (after filtering).
               Streaming stops fetching from HuggingFace as soon as this is met.
    """
    # rc.nocontext = reading-comprehension split, evidence passage stripped out
    # streaming=True: fetches parquet shards lazily — stops as soon as we have enough
    dataset = load_dataset(
        "trivia_qa",
        "rc.nocontext",
        split=split,
        trust_remote_code=True,
        streaming=True,
    )

    records = []
    for row in dataset:
        answer = row.get("answer", {})

        # normalized_aliases gives the broadest set of acceptable surface forms
        # (e.g. "usa", "united states", "u.s.a" all present for the same fact)
        aliases   = answer.get("normalized_aliases", [])
        norm_val  = answer.get("normalized_value", "").strip()

        # Build deduplicated answer list, primary value first
        correct_answers = list(
            dict.fromkeys(
                [a.strip() for a in ([norm_val] + aliases) if a.strip()]
            )
        )

        # Skip questions with no usable ground-truth answer
        if not correct_answers:
            continue

        records.append(
            {
                "question_id":    str(row["question_id"]),
                "question":       row["question"].strip(),
                "correct_answers": correct_answers,
            }
        )

        if limit and len(records) >= limit:
            break   # stops the HuggingFace stream immediately

    return records


def load_truthfulqa(split="validation", limit=None):
    """
    Load TruthfulQA (generation config) from HuggingFace.
    Returns a list of dicts:
        {
            "question_id":    str,   # derived from row index (no native ID field)
            "question":       str,
            "correct_answers": [str, ...]
        }

    TruthfulQA has 817 questions in the validation split, all adversarially
    designed to elicit confident-but-wrong answers from LLMs.

    Args:
        split: HuggingFace split — only "validation" exists for TruthfulQA.
        limit: if set, return at most this many questions.
    """
    dataset = load_dataset(
        "truthful_qa",
        "generation",
        split=split,
        trust_remote_code=True,
    )

    records = []
    for i, row in enumerate(dataset):
        # Merge best_answer + correct_answers, deduplicated
        best    = row.get("best_answer", "").strip()
        correct = [a.strip() for a in row.get("correct_answers", []) if a.strip()]
        correct_answers = list(dict.fromkeys([a for a in ([best] + correct) if a]))

        if not correct_answers:
            continue

        records.append(
            {
                "question_id":    f"tqa_{i}",
                "question":       row["question"].strip(),
                "correct_answers": correct_answers,
            }
        )

        if limit and len(records) >= limit:
            break

    return records
