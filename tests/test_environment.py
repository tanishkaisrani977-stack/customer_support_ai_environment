import unittest

from env.environment import CustomerSupportEnv
from env.models import Action


class CustomerSupportEnvTests(unittest.TestCase):
    def test_reset_returns_first_ticket_for_each_task(self):
        expected_ids = {
            "easy": 101,
            "medium": 201,
            "hard": 301,
        }

        for task_name, ticket_id in expected_ids.items():
            with self.subTest(task_name=task_name):
                env = CustomerSupportEnv(task_name=task_name)
                observation = env.reset()
                self.assertEqual(observation.ticket_id, ticket_id)
                self.assertEqual(env.state()["task_name"], task_name)
                self.assertGreater(env.state()["total_score"], 0.0)
                self.assertLess(env.state()["total_score"], 1.0)

    def test_easy_ticket_reaches_task_score_within_open_interval(self):
        env = CustomerSupportEnv(task_name="easy")
        env.reset()

        _, reward_1, done_1, _ = env.step(Action(action_type="classify", content="billing"))
        _, reward_2, done_2, info = env.step(
            Action(
                action_type="reply",
                content=(
                    "We are reviewing the billing issue and duplicate charge, processing the refund, "
                    "and will update you within 24 hours."
                ),
            )
        )

        self.assertEqual(reward_1.score, 0.4)
        self.assertEqual(reward_2.score, 0.6)
        self.assertFalse(done_1)
        self.assertTrue(done_2)
        self.assertGreater(info["total_score"], 0.0)
        self.assertLess(info["total_score"], 1.0)
        self.assertEqual(info["total_score"], 0.999)

    def test_hard_ticket_requires_multiple_actions_before_advancing(self):
        env = CustomerSupportEnv(task_name="hard")
        observation = env.reset()
        self.assertEqual(observation.ticket_id, 301)

        observation, _, done, info = env.step(Action(action_type="classify", content="technical"))
        self.assertFalse(done)
        self.assertEqual(info["current_ticket_id"], 301)

        observation, _, done, info = env.step(
            Action(
                action_type="reply",
                content=(
                    "We will investigate the access issue, review the payment signal, "
                    "keep your team updated, and prioritize access restoration."
                ),
            )
        )
        self.assertFalse(done)
        self.assertEqual(info["current_ticket_id"], 301)

        observation, reward, done, info = env.step(
            Action(
                action_type="escalate",
                content="Escalate to enterprise_support for urgent team access restoration.",
            )
        )
        self.assertFalse(done)
        self.assertEqual(reward.score, 0.2)
        self.assertEqual(info["current_ticket_id"], 302)
        self.assertEqual(observation.ticket_id, 302)

    def test_wrong_action_order_applies_penalty(self):
        env = CustomerSupportEnv(task_name="easy")
        env.reset()

        _, reward, done, info = env.step(Action(action_type="reply", content="We can help."))

        self.assertEqual(reward.score, 0.001)
        self.assertFalse(done)
        self.assertEqual(info["ticket_scores"], [0.001])
        self.assertGreater(info["total_score"], 0.0)
        self.assertLess(info["total_score"], 1.0)
        self.assertIn("Wrong action order penalty applied.", info["feedback"])


if __name__ == "__main__":
    unittest.main()
