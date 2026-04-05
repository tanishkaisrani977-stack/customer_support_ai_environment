from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from env.environment import CustomerSupportEnv
from env.models import Action
from env.tasks import list_tasks


class ResetRequest(BaseModel):
    task_name: str | None = None


class StepResponse(BaseModel):
    observation: dict
    reward: dict
    done: bool
    info: dict


app = FastAPI(title="OpenEnv Customer Support", version="1.0.0")
environment = CustomerSupportEnv()
ui_path = Path(__file__).resolve().parent / "ui" / "index.html"
openenv_manifest_path = Path(__file__).resolve().parent / "openenv.yaml"


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(ui_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def get_tasks():
    return {"tasks": list_tasks()}


@app.get("/metadata")
def get_metadata():
    return {
        "name": "openenv-customer-support",
        "version": "1.0.0",
        "runtime": "fastapi",
        "app": "server.app:app",
        "port": int(os.getenv("PORT", "8000")),
        "tasks": list_tasks(),
        "reward_range": [0.0, 1.0],
        "endpoints": {
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "health": "/health",
            "tasks": "/tasks",
            "schema": "/schema",
            "validate": "/validate",
        },
    }


@app.get("/schema")
def get_schema():
    return {
        "action": Action.model_json_schema(),
        "observation": environment.current_observation.model_json_schema(),
        "reward": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "feedback": {"type": "string"},
            },
            "required": ["score", "feedback"],
        },
    }


@app.get("/validate")
def validate_environment():
    manifest_exists = openenv_manifest_path.exists()
    return {
        "valid": manifest_exists,
        "checks": {
            "manifest_present": manifest_exists,
            "reset_endpoint": True,
            "step_endpoint": True,
            "state_endpoint": True,
        },
    }


@app.post("/reset")
def reset_environment(request: ResetRequest | None = None):
    if request and request.task_name:
        try:
            environment.select_task(request.task_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return environment.reset().model_dump()


@app.get("/state")
def get_state():
    return environment.state()


@app.post("/step", response_model=StepResponse)
def step_environment(action: Action):
    observation, reward, done, info = environment.step(action)
    return StepResponse(
        observation=observation.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


def main():
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
