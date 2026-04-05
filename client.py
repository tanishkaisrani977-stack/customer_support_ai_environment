from __future__ import annotations

import json
from typing import Any
from urllib import request

from env.models import Action, Observation


class CustomerSupportEnvClient:
    """Minimal typed HTTP client for reset/step/state interactions."""

    def __init__(self, base_url: str = "http://127.0.0.1:7860") -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_name: str | None = None) -> Observation:
        payload = {} if task_name is None else {"task_name": task_name}
        response = self._post_json("/reset", payload)
        return Observation.model_validate(response)

    def step(self, action: Action) -> dict[str, Any]:
        return self._post_json("/step", action.model_dump())

    def state(self) -> dict[str, Any]:
        return self._get_json("/state")

    def _get_json(self, path: str) -> dict[str, Any]:
        with request.urlopen(f"{self.base_url}{path}") as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request) as response:
            return json.loads(response.read().decode("utf-8"))
