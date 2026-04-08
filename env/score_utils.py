from __future__ import annotations

import math


EPS = 1e-6
SCORE_EPSILON = EPS
MAX_SAFE_SCORE = 1 - SCORE_EPSILON


def safe_score(score):
    epsilon = SCORE_EPSILON

    if score is None or (isinstance(score, float) and math.isnan(score)):
        return epsilon

    score = float(score)

    if score <= 0:
        score = epsilon
    elif score >= 1:
        score = 1 - epsilon

    assert 0 < score < 1, f"Invalid score: {score}"
    return score


def safe_ratio(correct, total):
    correct = 0.0 if correct is None else float(correct)
    total = 0.0 if total is None else float(total)
    if total <= 0:
        return safe_score(0.5)
    return safe_score((correct + SCORE_EPSILON) / (total + (2 * SCORE_EPSILON)))


def safe_score_list(scores):
    return [safe_score(score) for score in scores]


def validate_scores(scores):
    for name, score in scores.items():
        assert score is not None
        assert not (isinstance(score, float) and math.isnan(score))
        assert 0 < score < 1, f"{name} out of range: {score}"
