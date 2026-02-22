STOPWORDS = {"what", "does", "this", "that", "with", "from", "have", "your", "they", "when",
             "will", "been", "were", "there", "their", "about", "which", "would", "could"}

def label_response(response: str, correct_answers: list[str], question: str = "") -> int:
    """
    Returns 0 (factual) if the response overlaps with any correct answer.
    Three matching strategies (in order):
      1. Correct answer is a substring of the response
      2. Response is a substring of any correct answer
      3. Keyword overlap: enough meaningful words from the correct answer appear in the response
    """
    # Strip the echoed prompt from the response
    if question and response.lower().startswith(question.lower()):
        response = response[len(question):].strip()

    response_lower = response.lower().strip()
    if not response_lower:
        return 1

    for answer in correct_answers:
        answer_lower = answer.lower().strip()

        # Strategy 1: exact substring match (either direction)
        if answer_lower in response_lower or response_lower in answer_lower:
            return 0

        # Strategy 3: keyword overlap
        # Extract meaningful words (len > 3, not stopwords) from the correct answer
        keywords = [w for w in answer_lower.split()
                    if len(w) > 3 and w not in STOPWORDS]
        if keywords:
            matches = sum(1 for kw in keywords if kw in response_lower)
            # If more than half the keywords appear in the response → factual
            if matches / len(keywords) >= 0.5:
                return 0

    return 1


def majority_label(responses: list[str], correct_answers: list[str], question: str = "") -> int:
    """
    Labels each of the K responses and returns the majority vote.
    Tie → treated as hallucination (conservative).
    """
    labels = [label_response(r, correct_answers, question=question) for r in responses]
    hallucination_count = sum(labels)
    factual_count = len(labels) - hallucination_count
    return 0 if factual_count > hallucination_count else 1
