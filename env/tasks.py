from __future__ import annotations

from .models import ActionType, Priority, TaskDefinition, TicketCase, TicketCategory, TicketExpectation, TicketInput


TASKS: dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        name="easy",
        expected_outputs=[
            {
                "ticket_id": 101,
                "classification": "billing",
                "reply": "Acknowledge duplicate charge, billing review, and 24 hour follow-up.",
                "escalation": "not_required",
            }
        ],
        metadata={
            "difficulty": "easy",
            "description": "A single straightforward billing ticket with no escalation required.",
            "ticket_count": 1,
        },
        input_tickets=[
            TicketCase(
                input_ticket=TicketInput(
                    ticket_id=101,
                    user_message=(
                        "Hi, I was charged twice for order 4821 and I only placed one order. "
                        "Can you help me get the extra charge refunded?"
                    ),
                    history=["Customer opened ticket from the billing portal."],
                    priority=Priority.MEDIUM,
                ),
                expected_outputs=TicketExpectation(
                    classification=TicketCategory.BILLING,
                    required_reply_keywords=["refund", "duplicate charge", "billing", "24 hours"],
                    disallowed_reply_keywords=["ignore", "guarantee", "impossible", "fault"],
                    escalation_required=False,
                    escalation_team=None,
                    required_sequence=[ActionType.CLASSIFY, ActionType.REPLY],
                    max_actions=3,
                    expected_outputs={
                        "classification": "billing",
                        "reply": (
                            "Acknowledge the duplicate charge, explain the billing review, and set a 24 hour follow-up."
                        ),
                        "escalation": "not_required",
                    },
                ),
                metadata={
                    "scenario": "duplicate_charge",
                    "channel": "portal",
                    "customer_tier": "standard",
                },
            )
        ],
    ),
    "medium": TaskDefinition(
        name="medium",
        expected_outputs=[
            {
                "ticket_id": 201,
                "classification": "billing",
                "reply": "Confirm invoice review, annual plan correction, and billing resolution.",
                "escalation": "not_required",
            },
            {
                "ticket_id": 202,
                "classification": "account_access",
                "reply": "Explain identity verification, 2FA review, and secure phone update steps.",
                "escalation": "account_security",
            },
            {
                "ticket_id": 203,
                "classification": "technical",
                "reply": "Acknowledge export crash, recent update, and investigation steps.",
                "escalation": "not_required",
            },
        ],
        metadata={
            "difficulty": "medium",
            "description": "Multiple independent tickets spanning billing, account access, and technical support.",
            "ticket_count": 3,
        },
        input_tickets=[
            TicketCase(
                input_ticket=TicketInput(
                    ticket_id=201,
                    user_message=(
                        "I upgraded to the annual plan yesterday, but my invoice still shows the monthly amount twice. "
                        "Please fix the charge and confirm the correct billing cycle."
                    ),
                    history=["Upgrade completed at 2026-04-04 09:10 UTC."],
                    priority=Priority.MEDIUM,
                ),
                expected_outputs=TicketExpectation(
                    classification=TicketCategory.BILLING,
                    required_reply_keywords=["invoice", "annual plan", "refund", "billing"],
                    disallowed_reply_keywords=["impossible", "fault", "ignore", "guarantee"],
                    escalation_required=False,
                    escalation_team=None,
                    required_sequence=[ActionType.CLASSIFY, ActionType.REPLY],
                    max_actions=3,
                    expected_outputs={
                        "classification": "billing",
                        "reply": "Confirm invoice review, address annual plan pricing, and explain billing correction steps.",
                        "escalation": "not_required",
                    },
                ),
                metadata={"scenario": "plan_upgrade_billing", "channel": "email"},
            ),
            TicketCase(
                input_ticket=TicketInput(
                    ticket_id=202,
                    user_message=(
                        "I changed my phone number and now the login code keeps going to my old device. "
                        "I am locked out of the admin dashboard."
                    ),
                    history=["Customer already tried password reset once.", "Two-factor authentication is enabled."],
                    priority=Priority.HIGH,
                ),
                expected_outputs=TicketExpectation(
                    classification=TicketCategory.ACCOUNT_ACCESS,
                    required_reply_keywords=["verify", "2fa", "update phone", "secure"],
                    disallowed_reply_keywords=["share password", "disable security", "ignore", "guarantee"],
                    escalation_required=True,
                    escalation_team="account_security",
                    required_sequence=[ActionType.CLASSIFY, ActionType.REPLY, ActionType.ESCALATE],
                    max_actions=4,
                    expected_outputs={
                        "classification": "account_access",
                        "reply": "Explain identity verification, mention 2FA and secure phone update steps.",
                        "escalation": "account_security",
                    },
                ),
                metadata={"scenario": "2fa_lockout", "channel": "chat"},
            ),
            TicketCase(
                input_ticket=TicketInput(
                    ticket_id=203,
                    user_message=(
                        "The desktop app crashes every time I export a report after the latest update. "
                        "Restarting did not help."
                    ),
                    history=["Crash started after version 4.8.1 rollout."],
                    priority=Priority.MEDIUM,
                ),
                expected_outputs=TicketExpectation(
                    classification=TicketCategory.TECHNICAL,
                    required_reply_keywords=["export", "update", "restart", "investigate"],
                    disallowed_reply_keywords=["ignore", "impossible", "fault", "always"],
                    escalation_required=False,
                    escalation_team=None,
                    required_sequence=[ActionType.CLASSIFY, ActionType.REPLY],
                    max_actions=3,
                    expected_outputs={
                        "classification": "technical",
                        "reply": "Acknowledge the export crash, reference the update, and confirm investigation steps.",
                        "escalation": "not_required",
                    },
                ),
                metadata={"scenario": "post_release_bug", "channel": "portal"},
            ),
        ],
    ),
    "hard": TaskDefinition(
        name="hard",
        expected_outputs=[
            {
                "ticket_id": 301,
                "classification": "technical",
                "reply": "Address the access outage, payment investigation, urgency, and status updates.",
                "escalation": "enterprise_support",
            },
            {
                "ticket_id": 302,
                "classification": "shipping",
                "reply": "Explain carrier investigation, address correction review, replacement handling, and no extra charge.",
                "escalation": "logistics",
            },
        ],
        metadata={
            "difficulty": "hard",
            "description": "Ambiguous, high-priority tickets that require multiple actions on the same ticket.",
            "ticket_count": 2,
        },
        input_tickets=[
            TicketCase(
                input_ticket=TicketInput(
                    ticket_id=301,
                    user_message=(
                        "Our team subscription renewed this morning, then everyone was logged out, and one teammate says "
                        "the invoice shows a failed payment. We need access restored before payroll in two hours."
                    ),
                    history=[
                        "Workspace owner reports 18 affected users.",
                        "Status page shows no active outage.",
                    ],
                    priority=Priority.HIGH,
                ),
                expected_outputs=TicketExpectation(
                    classification=TicketCategory.TECHNICAL,
                    required_reply_keywords=["access", "investigate", "payment", "team", "update"],
                    disallowed_reply_keywords=["ignore", "guarantee", "fault", "wait indefinitely"],
                    escalation_required=True,
                    escalation_team="enterprise_support",
                    required_sequence=[ActionType.CLASSIFY, ActionType.REPLY, ActionType.ESCALATE],
                    max_actions=4,
                    expected_outputs={
                        "classification": "technical",
                        "reply": (
                            "Acknowledge the access outage, mention payment investigation, promise updates, and address urgency."
                        ),
                        "escalation": "enterprise_support",
                    },
                ),
                metadata={
                    "scenario": "ambiguous_access_and_billing",
                    "ambiguity": ["technical", "billing"],
                    "channel": "priority_email",
                },
            ),
            TicketCase(
                input_ticket=TicketInput(
                    ticket_id=302,
                    user_message=(
                        "I changed the shipping address yesterday, but tracking still shows the old city and the package "
                        "has not moved for six days. If it is lost I need a replacement, but I do not want to be charged twice."
                    ),
                    history=[
                        "Address edit requested after label creation.",
                        "Customer is concerned about duplicate charges.",
                    ],
                    priority=Priority.HIGH,
                ),
                expected_outputs=TicketExpectation(
                    classification=TicketCategory.SHIPPING,
                    required_reply_keywords=["carrier", "address", "replacement", "no extra charge", "investigate"],
                    disallowed_reply_keywords=["ignore", "guarantee", "fault", "wait indefinitely"],
                    escalation_required=True,
                    escalation_team="logistics",
                    required_sequence=[ActionType.CLASSIFY, ActionType.REPLY, ActionType.ESCALATE],
                    max_actions=4,
                    expected_outputs={
                        "classification": "shipping",
                        "reply": "Explain the carrier investigation, address the address issue, and reassure no duplicate charge.",
                        "escalation": "logistics",
                    },
                ),
                metadata={
                    "scenario": "stalled_shipment_with_billing_concern",
                    "ambiguity": ["shipping", "billing"],
                    "channel": "chat",
                },
            ),
        ],
    ),
}


def get_task(task_name: str) -> TaskDefinition:
    if task_name not in TASKS:
        available = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}")
    return TASKS[task_name].model_copy(deep=True)


def list_tasks() -> list[str]:
    return ["easy", "medium", "hard"]
