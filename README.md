---
title: OpenEnv Customer Support
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv Customer Support Environment

## Project Description

`openenv-customer-support` is a production-ready OpenEnv-style environment that simulates a real-world customer support AI workflow. An agent receives support tickets, chooses one action at a time, and is graded deterministically on how well it classifies the request, replies to the customer, and escalates when needed. The project is intentionally lightweight so it can run comfortably on CPU-only infrastructure with limited memory while still reflecting realistic support operations.

The repository is organized as a reusable environment package, a FastAPI service, an inference runner, and a lightweight test suite.

## Motivation

Customer support is a strong real-world benchmark for AI agents because it mixes structured reasoning with natural-language interaction. A capable support agent needs to:

- identify the underlying issue correctly
- produce a helpful, safe, and actionable reply
- know when escalation is required
- maintain state across multiple actions on the same ticket

This environment captures those constraints in a deterministic format that is useful for evaluation, demo environments, integration tests, and reinforcement-learning-style experimentation.

## Repository Structure

```text
openenv-customer-support/
|-- env/
|   |-- __init__.py
|   |-- environment.py
|   |-- grader.py
|   |-- models.py
|   `-- tasks.py
|-- tests/
|   |-- test_app.py
|   |-- test_environment.py
|   `-- test_inference.py
|-- .env.example
|-- app.py
|-- Dockerfile
|-- inference.py
|-- openenv.yaml
|-- README.md
|-- requirements.txt
`-- run.ps1
```

## Observation Space

Each environment reset or step returns an `Observation` Pydantic model with the following fields:

- `ticket_id` (`int`): unique identifier of the active support ticket
- `user_message` (`str`): the customer's current message
- `history` (`list[str]`): prior ticket history plus previously taken agent actions and grader feedback
- `priority` (`low | medium | high`): urgency of the current ticket

This observation format makes the environment stateful. The same ticket can remain active across multiple steps, and each new observation includes the interaction history gathered so far.

## Action Space

The environment accepts an `Action` Pydantic model:

- `action_type`: one of `classify`, `reply`, or `escalate`
- `content`: free-form string payload

Expected behavior:

- `classify`: assign a support category such as `billing` or `technical`
- `reply`: send a helpful customer-facing response
- `escalate`: route the ticket to the correct specialist team when necessary

## Reward Design

Rewards are deterministic and normalized to the range `0.0` to `1.0`.

### Scoring Components

- correct classification: `+0.4`
- helpful reply: up to `+0.4`
- correct escalation decision: `+0.2`

### Penalties

- wrong action order: `-0.1`
- duplicate action: `-0.05`
- wrong classification: `-0.2`
- unnecessary escalation: `-0.2`
- unsafe reply language: `-0.1` per unsafe phrase, capped deterministically
- weak reply that misses critical guidance: `-0.1`

### How Rewards Work

- The grader awards partial reply credit based on required keyword coverage.
- Penalties reduce the ticket's internal raw score.
- Final ticket scores are clipped to `0.0..1.0`.
- Episode score is the average of all ticket scores in the selected task.
- For tickets that should stay self-serve, the environment awards the escalation component when the ticket is completed without an unnecessary escalation.

## Task Descriptions

### Easy

One simple billing ticket:

- duplicate charge refund request
- clear billing classification
- helpful reply expected
- escalation not required

### Medium

Three independent tickets:

- billing correction after a plan upgrade
- account access / 2FA lockout that requires escalation to `account_security`
- technical export crash after an update

### Hard

Two ambiguous, high-priority tickets:

- a team subscription issue that blends access failure with billing signals and requires escalation to `enterprise_support`
- a stalled shipment with address and duplicate-charge concerns that requires escalation to `logistics`

These hard tickets are multi-step. The same ticket remains active across several `step()` calls until the required sequence is consumed or the ticket reaches its action budget.

## Environment API

The environment class is `CustomerSupportEnv` in `env/environment.py`.

### `reset() -> Observation`

Resets the selected task and returns the first ticket observation.

### `step(action) -> (observation, reward, done, info)`

Processes one `Action`, grades it, updates the history, advances to the next ticket when appropriate, and returns:

- next `Observation`
- `Reward`
- `done` flag
- `info` dictionary with progress and cumulative score

### `state() -> dict`

Returns a structured snapshot of:

- active task
- current ticket index
- current observation
- ticket progress
- total normalized score
- completion status

## FastAPI Service

The project includes a minimal API in `app.py`.

### Endpoints

- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`

Opening the root URL serves a tiny browser UI for resetting tasks, sending actions, and viewing live environment state.

### Reset Request Body

`POST /reset` optionally accepts:

```json
{
  "task_name": "easy"
}
```

If `task_name` is provided, the environment switches tasks before resetting.

### Step Request Body

`POST /step` accepts a typed action payload:

```json
{
  "action_type": "classify",
  "content": "billing"
}
```

The response contains:

- `observation`: the next observation snapshot
- `reward`: the scored reward object
- `done`: episode completion flag
- `info`: ticket progress and cumulative scoring details

## Setup Instructions

### Local Setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

If your Windows machine already has the required packages installed globally, this repo's `.venv` can inherit system site packages and still run the project successfully.

### Optional `.env` Configuration

Copy `.env.example` to `.env` and set:

```bash
OPENAI_API_KEY=your_api_key_here
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4.1-mini
```

`inference.py` automatically loads `.env` from the project root before reading environment variables.

### Run the FastAPI App

```bash
python app.py
```

The service starts on `http://0.0.0.0:8000`.

You can also use:

```powershell
.\run.ps1 -Action app
```

Then open:

```text
http://127.0.0.1:8000/
```

### Run With Docker

```bash
docker build -t openenv-customer-support .
docker run --rm -p 8000:8000 openenv-customer-support
```

## Hugging Face Spaces

This repo is prepared for a Docker Hugging Face Space.

Key deployment details:

- `README.md` includes Hugging Face Space metadata
- the app binds to `PORT` and defaults to `7860`
- the Docker image exposes port `7860`
- `.env`, virtual environments, logs, and temp files are excluded from Space uploads

To deploy:

1. Create a new Space on Hugging Face and choose `Docker`
2. Push this repository to the Space remote
3. Open the Space URL once the build finishes

## How to Run Inference

Set the required environment variables:

```bash
set OPENAI_API_KEY=your_api_key
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4.1-mini
```

Then run:

```bash
python inference.py
```

Or run:

```powershell
.\run.ps1 -Action inference
```

To fail fast on real API errors instead of using fallback:

```powershell
.\run.ps1 -Action inference-strict
```

The script:

- initializes the OpenAI client
- runs `easy`, `medium`, and `hard`
- validates the model output as an `Action`
- falls back to deterministic actions if parsing fails
- prints standardized logs for every task and step

## Helper Script

The repository includes `run.ps1` for common development tasks:

```powershell
.\run.ps1 -Action check
.\run.ps1 -Action app
.\run.ps1 -Action inference
.\run.ps1 -Action inference-strict
.\run.ps1 -Action tests
.\run.ps1 -Action all
```

`all` runs the unit tests first and then runs inference.

`check` validates `.env` and prints a redacted runtime config before you make API calls.

## Expected Output Format

`inference.py` prints logs in exactly this format:

```text
[START] Task=easy
[STEP] step=1 action={"action_type":"classify","content":"billing"} reward=0.400
[STEP] step=2 action={"action_type":"reply","content":"We are reviewing the billing issue..."} reward=0.600
[END] Task=easy TotalScore=1.000
```

The exact action content and reward values depend on the model output, but the line format remains fixed.

## openenv.yaml

The repository includes `openenv.yaml` with:

- environment name
- version
- task list
- reward range

This makes the project easy to register or inspect in tooling that expects a simple environment manifest.

## Performance and Deployment Notes

- designed for CPU-only execution
- small dependency footprint
- no external heavyweight frameworks beyond FastAPI, Pydantic, and OpenAI SDK
- suitable for 2 vCPU / 8 GB RAM environments

## Automated Tests

Run the full suite with:

```bash
python -m unittest discover -s tests -v
```

Or run:

```powershell
.\run.ps1 -Action tests
```

## Quick Smoke Test

After installing dependencies, you can manually inspect the environment with Python:

```python
from env.environment import CustomerSupportEnv
from env.models import Action

env = CustomerSupportEnv(task_name="easy")
obs = env.reset()
print(obs)
print(env.step(Action(action_type="classify", content="billing")))
print(env.state())
```
