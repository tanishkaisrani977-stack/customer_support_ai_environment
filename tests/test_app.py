import unittest

from fastapi.testclient import TestClient

from app import app, environment


class AppTests(unittest.TestCase):
    def setUp(self):
        environment.select_task("easy")
        environment.reset()
        self.client = TestClient(app)

    def test_state_endpoint_returns_snapshot(self):
        response = self.client.get("/state")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_name"], "easy")
        self.assertIn("current_observation", payload)
        self.assertIn("available_tasks", payload)

    def test_health_endpoint_returns_ok(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_tasks_endpoint_returns_all_tasks(self):
        response = self.client.get("/tasks")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tasks"], ["easy", "medium", "hard"])

    def test_metadata_endpoint_returns_manifest_details(self):
        response = self.client.get("/metadata")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "openenv-customer-support")
        self.assertEqual(payload["runtime"], "fastapi")
        self.assertIn("reset", payload["endpoints"])

    def test_schema_endpoint_returns_action_observation_and_reward_schemas(self):
        response = self.client.get("/schema")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("action", payload)
        self.assertIn("observation", payload)
        self.assertIn("reward", payload)

    def test_validate_endpoint_reports_valid_project(self):
        response = self.client.get("/validate")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["checks"]["manifest_present"])

    def test_root_serves_web_ui(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("OpenEnv Customer Support UI", response.text)

    def test_reset_endpoint_can_switch_tasks(self):
        response = self.client.post("/reset", json={"task_name": "hard"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ticket_id"], 301)
        self.assertEqual(payload["priority"], "high")

    def test_step_endpoint_returns_reward_and_progress(self):
        self.client.post("/reset", json={"task_name": "easy"})

        response = self.client.post(
            "/step",
            json={"action_type": "classify", "content": "billing"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["done"])
        self.assertGreater(payload["reward"]["score"], 0.0)
        self.assertLess(payload["reward"]["score"], 1.0)
        self.assertEqual(payload["info"]["current_ticket_id"], 101)
        self.assertEqual(payload["observation"]["ticket_id"], 101)

    def test_step_endpoint_completes_ticket(self):
        self.client.post("/reset", json={"task_name": "easy"})
        self.client.post("/step", json={"action_type": "classify", "content": "billing"})

        response = self.client.post(
            "/step",
            json={
                "action_type": "reply",
                "content": (
                    "We are reviewing the billing issue and duplicate charge, processing the refund, "
                    "and will update you within 24 hours."
                ),
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["done"])
        self.assertGreater(payload["reward"]["score"], 0.0)
        self.assertLess(payload["reward"]["score"], 1.0)
        self.assertIsNone(payload["info"]["current_ticket_id"])
        self.assertEqual(payload["observation"]["ticket_id"], -1)


if __name__ == "__main__":
    unittest.main()
