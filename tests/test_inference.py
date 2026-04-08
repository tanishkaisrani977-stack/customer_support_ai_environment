import io
import os
import shutil
import subprocess
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import inference
import httpx
from openai import RateLimitError


class DummyClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                raise RuntimeError("offline")


class InferenceTests(unittest.TestCase):
    @staticmethod
    def _parse_score(line: str, marker: str) -> float:
        return float(line.split(marker, 1)[1])

    def test_load_local_env_reads_dotenv_without_overwriting_existing_values(self):
        temp_dir = Path("tests") / f"tmp_env_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            env_path = temp_dir / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=file_key\nAPI_BASE_URL=http://localhost:9999/v1\nMODEL_NAME=file_model\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"MODEL_NAME": "existing_model"}, clear=False):
                inference._load_local_env(env_path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "file_key")
                self.assertEqual(os.environ["API_BASE_URL"], "http://localhost:9999/v1")
                self.assertEqual(os.environ["MODEL_NAME"], "existing_model")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_local_env_replaces_placeholder_key_and_normalizes_base_url(self):
        temp_dir = Path("tests") / f"tmp_env_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            env_path = temp_dir / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=real_key\nAPI_BASE_URL==https://api.openai.com/v1/\nMODEL_NAME=gpt-4.1-mini\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "your_api_key_here",
                    "API_BASE_URL": "=https://bad.example.com",
                },
                clear=False,
            ):
                inference._load_local_env(env_path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "real_key")
                self.assertEqual(os.environ["API_BASE_URL"], "https://api.openai.com/v1")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_runtime_config_rejects_placeholder_key(self):
        with patch.dict(
            os.environ,
            {
                "API_KEY": "your_api_key_here",
                "API_BASE_URL": "https://api.openai.com/v1",
                "MODEL_NAME": "gpt-4.1-mini",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "example value"):
                inference._get_runtime_config()

    def test_get_runtime_config_accepts_api_key_alias(self):
        with patch.dict(
            os.environ,
            {
                "API_KEY": "proxy_key",
                "OPENAI_API_KEY": "your_api_key_here",
                "API_BASE_URL": "https://proxy.example.com/v1",
                "MODEL_NAME": "proxy-model",
            },
            clear=False,
        ):
            api_key, base_url, model_name = inference._get_runtime_config()

            self.assertEqual(api_key, "proxy_key")
            self.assertEqual(base_url, "https://proxy.example.com/v1")
            self.assertEqual(model_name, "proxy-model")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "proxy_key")

    def test_get_runtime_config_uses_default_model_name(self):
        with patch.dict(
            os.environ,
            {
                "API_KEY": "proxy_key",
                "API_BASE_URL": "https://proxy.example.com/v1",
            },
            clear=True,
        ):
            _, _, model_name = inference._get_runtime_config()

            self.assertEqual(model_name, "gpt-4.1-mini")

    def test_get_runtime_config_rejects_invalid_base_url(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "real_key",
                "API_BASE_URL": "api.openai.com/v1",
                "MODEL_NAME": "gpt-4.1-mini",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "must start with http:// or https://"):
                inference._get_runtime_config()

    def test_run_task_prints_required_log_format(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            score = inference.run_task(DummyClient(), "offline-fallback", "easy")

        output = buffer.getvalue().strip().splitlines()
        self.assertEqual(output[0], "[START] Task=easy")
        self.assertTrue(output[1].startswith("[STEP] step=1 action="))
        self.assertGreater(self._parse_score(output[1], "reward="), 0.0)
        self.assertLess(self._parse_score(output[1], "reward="), 1.0)
        self.assertEqual(output[2], "[DEBUG] step=1 source=fallback reason=Inference request failed: RuntimeError: offline")
        self.assertTrue(output[3].startswith("[STEP] step=2 action="))
        self.assertGreater(self._parse_score(output[3], "reward="), 0.0)
        self.assertLess(self._parse_score(output[3], "reward="), 1.0)
        self.assertEqual(output[4], "[DEBUG] step=2 source=fallback reason=Inference request failed: RuntimeError: offline")
        self.assertTrue(output[5].startswith("[END] Task=easy TotalScore="))
        self.assertGreater(self._parse_score(output[5], "TotalScore="), 0.0)
        self.assertLess(self._parse_score(output[5], "TotalScore="), 1.0)
        self.assertGreater(score, 0.9)
        self.assertLess(score, 1.0)

    def test_main_runs_all_tasks(self):
        buffer = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "dummy",
                    "API_BASE_URL": "http://127.0.0.1:9/v1",
                    "MODEL_NAME": "offline-fallback",
                },
                clear=False,
            ),
            patch("inference._build_client", return_value=DummyClient()),
            redirect_stdout(buffer),
        ):
            inference.main()

        output = buffer.getvalue()
        self.assertIn("[START] Task=easy", output)
        self.assertIn("[START] Task=medium", output)
        self.assertIn("[START] Task=hard", output)
        self.assertIn("[END] Task=hard", output)
        self.assertIn("[DEBUG] step=1 source=fallback reason=Inference request failed: RuntimeError: offline", output)

    def test_main_runs_in_fallback_mode_when_env_vars_are_missing(self):
        buffer = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("inference._load_local_env", return_value=None),
            redirect_stdout(buffer),
        ):
            inference.main()

        output = buffer.getvalue()
        self.assertIn("[START] Task=easy", output)
        self.assertIn("[END] Task=hard", output)
        self.assertIn(
            "[DEBUG] step=1 source=fallback reason=Inference request failed: RuntimeError: OpenAI runtime config unavailable; using built-in fallback policy.",
            output,
        )

    def test_strict_mode_raises_instead_of_using_fallback(self):
        with patch.dict(os.environ, {"OPENENV_STRICT_INFERENCE": "1"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "Inference request failed: RuntimeError: offline"):
                inference._resolve_action(
                    DummyClient(),
                    "offline-fallback",
                    inference.Observation(
                        ticket_id=1,
                        user_message="My invoice is wrong.",
                        history=[],
                        priority="medium",
                    ),
                )

    def test_print_runtime_check_redacts_api_key(self):
        buffer = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-1234567890abcdef",
                    "API_BASE_URL": "https://api.openai.com/v1",
                    "MODEL_NAME": "gpt-4.1-mini",
                },
                clear=False,
            ),
            redirect_stdout(buffer),
        ):
            inference.print_runtime_check()

        output = buffer.getvalue()
        self.assertIn("Configuration check passed.", output)
        self.assertIn("OPENAI_API_KEY=sk-1...cdef", output)
        self.assertIn("API_BASE_URL=https://api.openai.com/v1", output)
        self.assertIn("MODEL_NAME=gpt-4.1-mini", output)

    def test_script_prints_clean_error_for_placeholder_key(self):
        repo_root = Path(__file__).resolve().parents[1]
        temp_dir = repo_root / "tests" / f"tmp_script_{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_root / "inference.py", temp_dir / "inference.py")
            shutil.copytree(repo_root / "env", temp_dir / "env")

            (temp_dir / ".env").write_text(
                "OPENAI_API_KEY=your_api_key_here\nAPI_BASE_URL=https://api.openai.com/v1\nMODEL_NAME=gpt-4.1-mini\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env.pop("API_KEY", None)
            env.pop("API_BASE_URL", None)
            env.pop("MODEL_NAME", None)
            env["OPENENV_STRICT_INFERENCE"] = "1"

            result = subprocess.run(
                ["python", "inference.py"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("[ERROR] Inference configuration unavailable: OPENAI_API_KEY is still set to the example value.", result.stderr)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_script_succeeds_without_openai_env_vars(self):
        repo_root = Path(__file__).resolve().parents[1]
        temp_dir = repo_root / "tests" / f"tmp_script_{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_root / "inference.py", temp_dir / "inference.py")
            shutil.copytree(repo_root / "env", temp_dir / "env")

            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env.pop("API_KEY", None)
            env.pop("API_BASE_URL", None)
            env.pop("MODEL_NAME", None)
            env.pop("OPENENV_STRICT_INFERENCE", None)

            result = subprocess.run(
                ["python", "inference.py"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("[WARN] Inference configuration unavailable: Missing required environment variable: API_KEY", result.stderr)
            self.assertIn("[START] Task=easy", result.stdout)
            self.assertIn("[END] Task=hard", result.stdout)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_format_inference_error_shortens_insufficient_quota(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(
            429,
            request=request,
            json={"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}},
        )
        error = RateLimitError(
            "Error code: 429 - {'error': {'message': 'You exceeded your current quota', 'code': 'insufficient_quota'}}",
            response=response,
            body={"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}},
        )

        message = inference._format_inference_error(error)

        self.assertIn("OpenAI quota exceeded for this API key or organization.", message)
        self.assertIn("billing/overview", message)

    def test_format_inference_error_for_missing_env_is_short(self):
        message = inference._format_inference_error(RuntimeError("Missing required environment variable: API_KEY"))

        self.assertEqual(
            message,
            "Inference configuration unavailable: Missing required environment variable: API_KEY",
        )


if __name__ == "__main__":
    unittest.main()
