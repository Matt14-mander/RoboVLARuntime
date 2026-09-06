from __future__ import annotations

import unittest

from robovla_runtime.core.types import ActionChunk, ActionSpec
from robovla_runtime.execution.action_buffer import ActionBuffer


def make_chunk(
    request_id: int = 1,
    episode_id: str = "episode-a",
    start_at: float = 0.0,
    count: int = 4,
) -> ActionChunk:
    spec = ActionSpec(1, 0.1)
    return ActionChunk(
        episode_id,
        request_id,
        2,
        start_at,
        start_at,
        start_at,
        spec,
        tuple((float(i),) for i in range(count)),
    )


class ActionBufferTests(unittest.TestCase):
    def test_each_action_is_consumed_at_most_once(self) -> None:
        buffer = ActionBuffer("episode-a")
        self.assertTrue(buffer.accept(make_chunk(), 0.0).accepted)
        first = buffer.pop(0.0)
        self.assertIsNotNone(first)
        self.assertEqual(first.action_index, 0)
        self.assertIsNone(buffer.pop(0.0))

    def test_accept_reports_expired_prefix(self) -> None:
        buffer = ActionBuffer("episode-a")
        result = buffer.accept(make_chunk(), 0.21)
        self.assertTrue(result.accepted)
        self.assertEqual(result.dropped_prefix_actions, 2)
        self.assertEqual(buffer.pop(0.21).action_index, 2)

    def test_rejects_expired_wrong_episode_and_duplicate(self) -> None:
        buffer = ActionBuffer("episode-a")
        self.assertEqual(
            buffer.accept(make_chunk(1, "episode-old"), 0.0).reason,
            "wrong_episode",
        )
        self.assertEqual(buffer.accept(make_chunk(2), 0.5).reason, "expired")
        self.assertTrue(buffer.accept(make_chunk(3), 0.0).accepted)
        self.assertEqual(buffer.accept(make_chunk(3), 0.0).reason, "duplicate")

    def test_coverage_uses_exclusive_chunk_end(self) -> None:
        buffer = ActionBuffer("episode-a")
        buffer.accept(make_chunk(), 0.0)
        self.assertAlmostEqual(buffer.coverage_s(0.15), 0.25)
        self.assertEqual(buffer.coverage_s(0.4), 0.0)

    def test_execution_horizon_limits_accepted_chunk(self) -> None:
        buffer = ActionBuffer("episode-a", max_actions_per_chunk=2)
        buffer.accept(make_chunk(count=4), 0.0)
        self.assertAlmostEqual(buffer.coverage_s(0.0), 0.2)
        self.assertIsNone(buffer.pop(0.2))

    def test_reset_rejects_late_chunk_from_previous_episode(self) -> None:
        buffer = ActionBuffer("episode-a")
        old_chunk = make_chunk(request_id=4, episode_id="episode-a")
        buffer.reset("episode-b")
        result = buffer.accept(old_chunk, 0.0)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "wrong_episode")
        self.assertIsNone(buffer.pop(0.0))


if __name__ == "__main__":
    unittest.main()
