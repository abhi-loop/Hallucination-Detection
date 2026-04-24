"""
data/loader.py

Loads the TriviaQA dataset from HuggingFace (open-domain, no context).
Only questions with at least one non-empty answer alias are kept.

Uses streaming=True so only the records actually needed are downloaded —
no full-corpus download required even for small --limit runs.
"""

from datasets import load_dataset


def load_tqa(split="validation", limit=None):
    """
    Load TriviaQA (rc.nocontext) from HuggingFace using streaming mode.
    Returns a list of dicts:
        {
            "question_id": str,
            "question":    str,
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
