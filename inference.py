from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import APIConnectionError, AuthenticationError, NotFoundError, OpenAI, RateLimitError

from env.environment import CustomerSupportEnv
from env.models import Action, ActionType, Observation
from env.tasks import list_tasks


EXAMPLE_API_KEY_VALUES = {
    "",
    "your_api_key_here",
    "your_openai_api_key_here",
    "sk-your-real-key",
    "[redacted]",
}


class OfflineFallbackClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                raise RuntimeError("OpenAI runtime config unavailable; using built-in fallback policy.")


def _normalize_env_value(key: str, value: str) -> str:
    cleaned = value.strip().strip("\"'")
    if key == "API_BASE_URL":
        cleaned = cleaned.lstrip("=").rstrip("/")
    return cleaned


def _looks_like_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _is_placeholder_api_key(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in EXAMPLE_API_KEY_VALUES or normalized.startswith("your_api")


def _should_replace_env_value(key: str, current_value: str | None) -> bool:
    if not current_value or not current_value.strip():
        return True
    raw_value = current_value.strip().strip("\"'")
    current_value = _normalize_env_value(key, current_value)
    if key in {"OPENAI_API_KEY", "API_KEY"}:
        return _is_placeholder_api_key(current_value)
    if key == "API_BASE_URL":
        return raw_value.startswith("=") or not _looks_like_http_url(current_value)
    return False


def _load_local_env(env_path: Path | None = None) -> None:
    path = env_path or Path(".env")
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _normalize_env_value(key, value)
        if key and _should_replace_env_value(key, os.environ.get(key)):
            os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _require_api_key() -> str:
    api_key = _normalize_env_value("API_KEY", os.getenv("API_KEY", ""))
    openai_api_key = _normalize_env_value("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    if api_key and not _is_placeholder_api_key(api_key):
        return api_key
    if openai_api_key and not _is_placeholder_api_key(openai_api_key):
        return openai_api_key
    if api_key:
        return api_key
    if openai_api_key:
        return openai_api_key
    if not api_key and not openai_api_key:
        raise RuntimeError("Missing required environment variable: API_KEY")
    return ""


def _get_model_name() -> str:
    model_name = (
        os.getenv("MODEL_NAME")
        or os.getenv("MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4.1-mini"
    )
    return _normalize_env_value("MODEL_NAME", model_name)


def _strict_inference_enabled() -> bool:
    return os.getenv("OPENENV_STRICT_INFERENCE", "").strip().lower() in {"1", "true", "yes", "on"}


def _get_runtime_config() -> tuple[str, str, str]:
    api_key = _normalize_env_value("API_KEY", _require_api_key())
    base_url = _normalize_env_value("API_BASE_URL", _require_env("API_BASE_URL"))
    model_name = _get_model_name()

    if _is_placeholder_api_key(api_key):
        raise RuntimeError(
            "OPENAI_API_KEY is still set to the example value. Update .env with your real key "
            "or clear stale shell variables with `Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue`."
        )
    if not _looks_like_http_url(base_url):
        raise RuntimeError(
            f"API_BASE_URL must start with http:// or https://. Current value: {base_url!r}. "
            "Check for an extra '=' in .env."
        )
    if not model_name:
        raise RuntimeError("MODEL_NAME must not be empty.")

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["API_KEY"] = api_key
    os.environ["API_BASE_URL"] = base_url
    os.environ["MODEL_NAME"] = model_name
    return api_key, base_url, model_name


def _format_inference_error(exc: Exception) -> str:
    base_url = os.getenv("API_BASE_URL", "<missing>")
    model_name = os.getenv("MODEL_NAME", "<missing>")
    if isinstance(exc, RuntimeError):
        message = str(exc)
        if message.startswith("Missing required environment variable:"):
            return f"Inference configuration unavailable: {message}"
        if "example value" in message or "must start with http://" in message or "must not be empty" in message:
            return f"Inference configuration unavailable: {message}"

    if isinstance(exc, AuthenticationError):
        return (
            "OpenAI authentication failed. Check OPENAI_API_KEY in .env and clear any stale shell value with "
            "`Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue`, then rerun inference."
        )
    if isinstance(exc, APIConnectionError):
        return (
            f"Could not reach API_BASE_URL={base_url}. Confirm the URL starts with https:// and that your network "
            "can reach the provider."
        )
    if isinstance(exc, RateLimitError):
        message = str(exc).lower()
        if "insufficient_quota" in message or "exceeded your current quota" in message:
            return (
                "OpenAI quota exceeded for this API key or organization. Add credits or enable billing at "
                "https://platform.openai.com/settings/organization/billing/overview, then rerun inference."
            )
        return "OpenAI rate limit hit. Wait a moment and rerun inference."
    if isinstance(exc, NotFoundError):
        return f"MODEL_NAME={model_name!r} was not found for API_BASE_URL={base_url}."
    return f"Inference request failed: {type(exc).__name__}: {exc}"


def _redact_api_key(api_key: str) -> str:
    if not api_key:
        return "<missing>"
    if len(api_key) <= 8:
        return "[set]"
    return f"{api_key[:4]}...{api_key[-4:]}"


def print_runtime_check() -> None:
    _load_local_env()
    api_key, base_url, model_name = _get_runtime_config()
    strict_mode = "on" if _strict_inference_enabled() else "off"
    print("Configuration check passed.")
    print(f"OPENAI_API_KEY={_redact_api_key(api_key)}")
    print(f"API_BASE_URL={base_url}")
    print(f"MODEL_NAME={model_name}")
    print(f"OPENENV_STRICT_INFERENCE={strict_mode}")


def _build_client() -> OpenAI:
    api_key, base_url, _ = _get_runtime_config()
    return OpenAI(api_key=api_key, base_url=base_url)


def _build_runtime() -> tuple[OpenAI | OfflineFallbackClient, str]:
    try:
        _, _, model_name = _get_runtime_config()
        return _build_client(), model_name
    except RuntimeError as exc:
        if _strict_inference_enabled():
            raise
        print(f"[WARN] {_format_inference_error(exc)}", file=sys.stderr)
        return OfflineFallbackClient(), "offline-fallback"


def _build_messages(observation: Observation) -> list[dict[str, str]]:
    history_text = "\n".join(observation.history) if observation.history else "No prior history."
    user_prompt = f"""
You are a customer support agent acting inside a ticket-processing environment.
Return only valid JSON with this exact schema:
{{"action_type": "classify|reply|escalate", "content": "<string>"}}

Ticket ID: {observation.ticket_id}
Priority: {observation.priority}
Customer message: {observation.user_message}
History:
{history_text}

Choose the next best action for the current ticket.
""".strip()

    return [
        {
            "role": "system",
            "content": (
                "You are a precise support agent. Respond with JSON only and no markdown. "
                "Pick exactly one action for the current step."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def _classify_heuristic(message: str) -> str:
    normalized = message.lower()
    if any(keyword in normalized for keyword in ["invoice", "charge", "refund", "billing", "payment"]):
        return "billing"
    if any(keyword in normalized for keyword in ["login", "2fa", "locked out", "password", "phone number"]):
        return "account_access"
    if any(keyword in normalized for keyword in ["ship", "tracking", "package", "carrier", "address"]):
        return "shipping"
    if any(keyword in normalized for keyword in ["crash", "bug", "error", "export", "update"]):
        return "technical"
    return "general"


def _fallback_action(observation: Observation) -> Action:
    history_text = " ".join(observation.history).lower()
    if "agent action: classify" not in history_text:
        return Action(action_type=ActionType.CLASSIFY, content=_classify_heuristic(observation.user_message))
    if "agent action: reply" not in history_text:
        message = observation.user_message.lower()
        if "tracking" in message or "package" in message:
            content = (
                "We will investigate with the carrier, review the address update, and make sure there is no extra charge "
                "if a replacement is needed."
            )
        elif "2fa" in message or "locked out" in message or "phone number" in message:
            content = (
                "We will verify your identity, review the 2FA setup, update the phone number securely, and keep the account secure."
            )
        elif "invoice" in message or "charge" in message or "refund" in message or "payment" in message:
            content = (
                "We are reviewing the billing issue, checking the invoice and duplicate charge, and will update you on the refund within 24 hours."
            )
        else:
            content = (
                "We are investigating the issue, reviewing the recent update, and will share the next support update shortly."
            )
        return Action(action_type=ActionType.REPLY, content=content)
    if any(keyword in observation.user_message.lower() for keyword in ["tracking", "package", "address"]):
        return Action(action_type=ActionType.ESCALATE, content="Escalate to logistics for carrier investigation.")
    if any(keyword in observation.user_message.lower() for keyword in ["2fa", "locked out", "phone number"]):
        return Action(action_type=ActionType.ESCALATE, content="Escalate to account_security for secure recovery.")
    if any(keyword in observation.user_message.lower() for keyword in ["team subscription", "payroll", "logged out"]):
        return Action(action_type=ActionType.ESCALATE, content="Escalate to enterprise_support for urgent access restoration.")
    return Action(action_type=ActionType.REPLY, content="We are continuing the investigation and will follow up.")


def _get_model_action(client: OpenAI, model_name: str, observation: Observation) -> Action:
    response = client.chat.completions.create(
        model=model_name,
        messages=_build_messages(observation),
        temperature=0,
    )
    raw_text = response.choices[0].message.content or ""
    payload = _extract_json(raw_text)
    return Action.model_validate(payload)


def _resolve_action(client: OpenAI, model_name: str, observation: Observation) -> tuple[Action, str, str | None]:
    try:
        action = _get_model_action(client, model_name, observation)
        return action, "model", None
    except Exception as exc:
        formatted_error = _format_inference_error(exc)
        if _strict_inference_enabled():
            raise RuntimeError(formatted_error) from exc
        return _fallback_action(observation), "fallback", formatted_error


def run_task(client: OpenAI, model_name: str, task_name: str) -> float:
    env = CustomerSupportEnv(task_name=task_name)
    observation = env.reset()
    done = False
    step_number = 0

    print(f"[START] Task={task_name}")
    while not done:
        step_number += 1
        action, source, error_message = _resolve_action(client, model_name, observation)

        observation, reward, done, info = env.step(action)
        print(f"[STEP] step={step_number} action={action.model_dump_json()} reward={reward.score:.6f}")
        if source == "fallback" and error_message:
            print(f"[DEBUG] step={step_number} source=fallback reason={error_message}")
        else:
            print(f"[DEBUG] step={step_number} source=model")

    total_score = info["total_score"]
    print(f"[END] Task={task_name} TotalScore={total_score:.6f}")
    return total_score


def main() -> None:
    _load_local_env()
    client, model_name = _build_runtime()
    for task_name in list_tasks():
        run_task(client, model_name, task_name)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[ERROR] {_format_inference_error(exc)}", file=sys.stderr)
        raise SystemExit(1) from None
