
from __future__ import annotations

import math

# We use 0.01 instead of 1e-6 to ensure the platform 
# doesn't round the value back to 0.0 or 1.0.
EPS = 0.01
SCORE_EPSILON = EPS
MAX_SAFE_SCORE = 1 - SCORE_EPSILON


def safe_score(score):
    epsilon = SCORE_EPSILON

    # Handle None or NaN values immediately
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return epsilon

    try:
        score = float(score)
    except (ValueError, TypeError):
        return epsilon

    # Strict clamping logic:
    # If the score is 0 or less, it becomes 0.01
    # If the score is 1 or more, it becomes 0.99
    # This ensures it is ALWAYS strictly between 0 and 1.
    if score <= 0:
        score = epsilon
    elif score >= 1:
        score = 1 - epsilon
    else:
        # Extra safety check for values very close to the edges
        score = max(epsilon, min(1 - epsilon, score))

    assert 0 < score < 1, f"Invalid score: {score}"
    return score


def safe_ratio(correct, total):
    correct = 0.0 if correct is None else float(correct)
    total = 0.0 if total is None else float(total)
    
    if total <= 0:
        return safe_score(0.5)
    
    # Calculate raw ratio and let safe_score handle the bounds
    return safe_score(correct / total)


def safe_score_list(scores):
    return [safe_score(score) for score in scores]


def validate_scores(scores):
    for name, score in scores.items():
        assert score is not None
        assert not (isinstance(score, float) and math.isnan(score))
        # The validator at Phase 2 requires 0 < score < 1
        assert 0 < score < 1, f"{name} out of range: {score}"