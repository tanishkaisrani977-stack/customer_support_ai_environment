from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from env.environment import CustomerSupportEnv
from env.models import Action


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


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(ui_path)


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
