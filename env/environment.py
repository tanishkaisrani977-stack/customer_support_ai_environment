from __future__ import annotations

from .grader import DeterministicGrader
from .models import Action, EnvironmentSnapshot, Observation, Priority, Reward, TaskDefinition, TicketCase, TicketProgress
from .score_utils import SCORE_EPSILON, safe_score, safe_score_list, validate_scores
from .tasks import get_task, list_tasks


class CustomerSupportEnv:
    def __init__(self, task_name: str = "easy") -> None:
        self._grader = DeterministicGrader()
        self.available_tasks = list_tasks()
        self.task_name = task_name
        self.task: TaskDefinition = get_task(task_name)
        self.current_ticket_index = 0
        self.step_count = 0
        self.done = False
        self.ticket_progress: list[TicketProgress] = []
        self.total_score = 0.0
        self.emitted_reward_total = 0.0
        self.current_observation = self._terminal_observation()
        self.reset()

    def select_task(self, task_name: str) -> None:
        self.task_name = task_name
        self.task = get_task(task_name)

    def reset(self) -> Observation:
        self.task = get_task(self.task_name)
        self.current_ticket_index = 0
        self.step_count = 0
        self.done = False
        self.ticket_progress = [TicketProgress() for _ in self.task.input_tickets]
        self.total_score = safe_score(0.0)
        self.emitted_reward_total = 0.0
        self.current_observation = self._build_observation(self.current_ticket_index)
        return self.current_observation

    def step(self, action: Action | dict) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            reward = Reward(score=safe_score(0.0), feedback="Episode already complete.")
            return self.current_observation, reward, True, self._build_info(False, True, reward.feedback)

        parsed_action = action if isinstance(action, Action) else Action.model_validate(action)
        ticket_case = self.task.input_tickets[self.current_ticket_index]
        progress = self.ticket_progress[self.current_ticket_index]

        self.step_count += 1
        progress.action_count += 1
        progress.action_history.append(
            f"Agent action: {parsed_action.action_type} -> {parsed_action.content}"
        )

        grade_result = self._grader.evaluate_action(ticket_case, parsed_action, progress)
        progress.action_history.append(f"System feedback: {grade_result.feedback}")

        ticket_completed = self._should_complete_ticket(ticket_case, progress)
        advanced = False

        if ticket_completed:
            progress.completed = True
            advanced = self._advance_ticket()

        self.total_score = self._compute_total_score()
        self.current_observation = (
            self._terminal_observation() if self.done else self._build_observation(self.current_ticket_index)
        )
        reward = self._build_task_reward(grade_result.feedback)
        info = self._build_info(advanced, ticket_completed, grade_result.feedback)
        return self.current_observation, reward, self.done, info

    def state(self) -> dict:
        snapshot = EnvironmentSnapshot(
            task_name=self.task_name,
            current_ticket_index=self.current_ticket_index,
            step_count=self.step_count,
            total_score=safe_score(self.total_score),
            done=self.done,
            current_observation=self.current_observation,
            ticket_progress=self.ticket_progress,
        )
        payload = snapshot.model_dump()
        payload["total_score"] = safe_score(payload["total_score"])
        for progress in payload["ticket_progress"]:
            progress["raw_score"] = safe_score(progress["raw_score"])
        validate_scores({"total_score": payload["total_score"]})
        validate_scores(
            {f"ticket_progress_{index}_raw_score": progress["raw_score"] for index, progress in enumerate(payload["ticket_progress"])}
        )
        payload["available_tasks"] = self.available_tasks
        payload["task_metadata"] = self.task.metadata
        payload["task_expected_outputs"] = self.task.expected_outputs
        return payload

    def _should_complete_ticket(self, ticket_case: TicketCase, progress: TicketProgress) -> bool:
        required_steps = len(ticket_case.expected_outputs.required_sequence)
        exhausted = progress.action_count >= ticket_case.expected_outputs.max_actions
        return len(progress.consumed_actions) >= required_steps or exhausted

    def _advance_ticket(self) -> bool:
        if self.current_ticket_index + 1 >= len(self.task.input_tickets):
            self.done = True
            return False

        self.current_ticket_index += 1
        return True

    def _compute_total_score(self) -> float:
        if not self.ticket_progress:
            return safe_score(0.0)
        ticket_scores = [safe_score(progress.raw_score) for progress in self.ticket_progress]
        final_score = (sum(ticket_scores) + SCORE_EPSILON) / (float(len(ticket_scores)) + (2 * SCORE_EPSILON))
        final_score = safe_score(final_score)
        assert 0 < final_score < 1, f"Invalid score: {final_score}"
        return final_score

    def _normalize_task_score(self, value: float) -> float:
        return safe_score(value)

    def _build_task_reward(self, feedback: str) -> Reward:
        if self.done:
            remaining_score = float(self.total_score) - float(self.emitted_reward_total)
            reward_score = remaining_score if remaining_score > SCORE_EPSILON else SCORE_EPSILON
        else:
            reward_score = SCORE_EPSILON

        if self.done:
            reward_score = min(float(self.total_score), float(reward_score))
        reward_score = safe_score(reward_score)
        self.emitted_reward_total = min(float(self.total_score), float(self.emitted_reward_total + reward_score))
        assert 0 < reward_score < 1, f"Invalid score: {reward_score}"
        return Reward(score=reward_score, feedback=feedback)

    def _build_observation(self, ticket_index: int) -> Observation:
        ticket_case = self.task.input_tickets[ticket_index]
        progress = self.ticket_progress[ticket_index]
        history = ticket_case.input_ticket.history + progress.action_history
        return Observation(
            ticket_id=ticket_case.input_ticket.ticket_id,
            user_message=ticket_case.input_ticket.user_message,
            history=history,
            priority=ticket_case.input_ticket.priority,
        )

    def _terminal_observation(self) -> Observation:
        return Observation(
            ticket_id=-1,
            user_message="All tickets processed.",
            history=[],
            priority=Priority.LOW,
        )

    def _build_info(self, advanced: bool, ticket_completed: bool, feedback: str) -> dict:
        current_ticket_id = (
            self.task.input_tickets[self.current_ticket_index].input_ticket.ticket_id if not self.done else None
        )
        ticket_scores = safe_score_list(float(progress.raw_score) for progress in self.ticket_progress)
        total_score = safe_score(self.total_score)
        validate_scores({"total_score": total_score, **{f"ticket_score_{index}": score for index, score in enumerate(ticket_scores)}})
        return {
            "task_name": self.task_name,
            "step_count": self.step_count,
            "current_ticket_index": self.current_ticket_index,
            "current_ticket_id": current_ticket_id,
            "ticket_completed": ticket_completed,
            "advanced_to_next_ticket": advanced,
            "ticket_scores": ticket_scores,
            "total_score": total_score,
            "done": self.done,
            "feedback": feedback,
        }
