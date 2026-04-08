from __future__ import annotations

from dataclasses import dataclass

from .models import Action, ActionType, Reward, TicketCase, TicketProgress
from .score_utils import SCORE_EPSILON, safe_ratio, safe_score


CLASSIFICATION_SCORE = 0.4
REPLY_SCORE = 0.4
ESCALATION_SCORE = 0.2
WRONG_ORDER_PENALTY = 0.1
DUPLICATE_ACTION_PENALTY = 0.05
WRONG_CLASSIFICATION_PENALTY = 0.2
WRONG_ESCALATION_PENALTY = 0.2
UNSAFE_REPLY_PENALTY = 0.1
WEAK_REPLY_PENALTY = 0.1
REWARD_SCALE = 0.999


@dataclass
class GradeResult:
    reward: Reward
    raw_delta: float
    feedback: str


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _clamp_score(value: float) -> float:
    return safe_score(value)


def _reward_score(value: float) -> float:
    return _clamp_score(float(value) * REWARD_SCALE)


def _increment_score(current_score: float, delta: float) -> float:
    return safe_score(float(current_score) + float(delta))


def _nonzero_reward_base(value: float) -> float:
    if value is None or float(value) <= 0:
        return SCORE_EPSILON
    return float(value)


class DeterministicGrader:
    def evaluate_action(self, ticket_case: TicketCase, action: Action, progress: TicketProgress) -> GradeResult:
        expectation = ticket_case.expected_outputs
        action_type = action.action_type
        normalized_content = _normalize(action.content)
        expected_classification = str(expectation.classification)
        feedback_parts: list[str] = []
        raw_delta = 0.0

        if action_type == ActionType.ESCALATE:
            progress.escalation_attempted = True

        if action_type in progress.consumed_actions:
            progress.raw_score = _increment_score(progress.raw_score, -DUPLICATE_ACTION_PENALTY)
            progress.penalties += DUPLICATE_ACTION_PENALTY
            feedback = "Duplicate action penalty applied."
            score = _reward_score(SCORE_EPSILON)
            assert 0 < score < 1, f"Invalid score: {score}"
            return GradeResult(
                reward=Reward(score=score, feedback=feedback),
                raw_delta=-DUPLICATE_ACTION_PENALTY,
                feedback=feedback,
            )

        expected_index = len(progress.consumed_actions)
        expected_action = (
            expectation.required_sequence[expected_index]
            if expected_index < len(expectation.required_sequence)
            else None
        )

        if expected_action is None or action_type != expected_action:
            progress.raw_score = _increment_score(progress.raw_score, -WRONG_ORDER_PENALTY)
            progress.penalties += WRONG_ORDER_PENALTY
            feedback = "Wrong action order penalty applied."
            score = _reward_score(SCORE_EPSILON)
            assert 0 < score < 1, f"Invalid score: {score}"
            return GradeResult(
                reward=Reward(score=score, feedback=feedback),
                raw_delta=-WRONG_ORDER_PENALTY,
                feedback=feedback,
            )

        progress.consumed_actions.append(action_type)

        if action_type == ActionType.CLASSIFY:
            if expected_classification in normalized_content:
                raw_delta += CLASSIFICATION_SCORE
                progress.correct_actions.append(ActionType.CLASSIFY)
                feedback_parts.append("Correct classification.")
            else:
                raw_delta -= WRONG_CLASSIFICATION_PENALTY
                progress.penalties += WRONG_CLASSIFICATION_PENALTY
                feedback_parts.append(
                    f"Incorrect classification. Expected {expected_classification}."
                )

        elif action_type == ActionType.REPLY:
            required_keywords = expectation.required_reply_keywords
            matched_keywords = [keyword for keyword in required_keywords if keyword in normalized_content]
            coverage_ratio = safe_ratio(len(matched_keywords), len(required_keywords)) if required_keywords else (1.0 - SCORE_EPSILON)
            helpful_score = REPLY_SCORE * coverage_ratio
            unsafe_hits = [keyword for keyword in expectation.disallowed_reply_keywords if keyword in normalized_content]
            unsafe_penalty = min(len(unsafe_hits), 2) * UNSAFE_REPLY_PENALTY

            raw_delta += helpful_score
            if unsafe_penalty:
                raw_delta -= unsafe_penalty
                progress.penalties += unsafe_penalty
                feedback_parts.append(f"Unsafe reply penalty for: {', '.join(unsafe_hits)}.")

            if coverage_ratio < 0.5:
                raw_delta -= WEAK_REPLY_PENALTY
                progress.penalties += WEAK_REPLY_PENALTY
                feedback_parts.append("Reply missed critical guidance.")
            else:
                progress.correct_actions.append(ActionType.REPLY)
                feedback_parts.append(
                    f"Helpful reply matched {len(matched_keywords)}/{len(required_keywords)} required points."
                )

        elif action_type == ActionType.ESCALATE:
            if not expectation.escalation_required:
                raw_delta -= WRONG_ESCALATION_PENALTY
                progress.penalties += WRONG_ESCALATION_PENALTY
                feedback_parts.append("Escalation was not required for this ticket.")
            else:
                route_match = expectation.escalation_team and expectation.escalation_team.lower() in normalized_content
                if route_match:
                    raw_delta += ESCALATION_SCORE
                    progress.correct_actions.append(ActionType.ESCALATE)
                    progress.escalation_decision_awarded = True
                    feedback_parts.append("Correct escalation route selected.")
                else:
                    raw_delta += ESCALATION_SCORE / 2
                    feedback_parts.append(
                        f"Escalation attempted without the expected route {expectation.escalation_team}."
                    )

        progress.raw_score = _increment_score(progress.raw_score, raw_delta)

        if (
            not expectation.escalation_required
            and len(progress.consumed_actions) == len(expectation.required_sequence)
            and not progress.escalation_decision_awarded
            and not progress.escalation_attempted
        ):
            progress.raw_score = _increment_score(progress.raw_score, ESCALATION_SCORE)
            raw_delta += ESCALATION_SCORE
            progress.escalation_decision_awarded = True
            feedback_parts.append("Correctly kept the ticket self-serve without escalation.")

        feedback = " ".join(feedback_parts) or "Action evaluated."
        score = _reward_score(_nonzero_reward_base(raw_delta))
        assert 0 < progress.raw_score < 1, f"Invalid score: {progress.raw_score}"
        assert 0 < score < 1, f"Invalid score: {score}"
        return GradeResult(
            reward=Reward(score=score, feedback=feedback),
            raw_delta=float(raw_delta),
            feedback=feedback,
        )
