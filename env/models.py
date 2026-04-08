from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StringEnum(str, Enum):
    pass


class Priority(StringEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionType(StringEnum):
    CLASSIFY = "classify"
    REPLY = "reply"
    ESCALATE = "escalate"


class TicketCategory(StringEnum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT_ACCESS = "account_access"
    SHIPPING = "shipping"
    GENERAL = "general"


class Observation(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    ticket_id: int
    user_message: str
    history: list[str] = Field(default_factory=list)
    priority: Priority


class Action(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    action_type: ActionType
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Action content must not be empty.")
        return cleaned


class Reward(BaseModel):
    score: float = Field(ge=1e-6, le=1 - 1e-6)
    feedback: str


class TicketInput(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    ticket_id: int
    user_message: str
    history: list[str] = Field(default_factory=list)
    priority: Priority


class TicketExpectation(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    classification: TicketCategory
    required_reply_keywords: list[str] = Field(default_factory=list)
    disallowed_reply_keywords: list[str] = Field(default_factory=list)
    escalation_required: bool = False
    escalation_team: str | None = None
    required_sequence: list[ActionType] = Field(default_factory=list)
    max_actions: int = 3
    expected_outputs: dict[str, Any] = Field(default_factory=dict)


class TicketCase(BaseModel):
    input_ticket: TicketInput
    expected_outputs: TicketExpectation
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskDefinition(BaseModel):
    name: str
    input_tickets: list[TicketCase]
    expected_outputs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TicketProgress(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    consumed_actions: list[ActionType] = Field(default_factory=list)
    correct_actions: list[ActionType] = Field(default_factory=list)
    action_history: list[str] = Field(default_factory=list)
    raw_score: float = 0.0
    penalties: float = 0.0
    action_count: int = 0
    escalation_attempted: bool = False
    escalation_decision_awarded: bool = False
    completed: bool = False


class EnvironmentSnapshot(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    task_name: str
    current_ticket_index: int
    step_count: int
    total_score: float
    done: bool
    current_observation: Observation
    ticket_progress: list[TicketProgress]
