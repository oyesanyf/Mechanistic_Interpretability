#!/usr/bin/env python3
"""
Unit tests for MCTSNode, PUCT selection, expansion, backpropagation, and MonteCarloTreeSearch.
"""

import unittest
from rest_rl.mcts import MCTSNode, MCTSConfig, MonteCarloTreeSearch, MCTSTrace


class TestMCTSSearch(unittest.TestCase):
    def test_node_puct_and_trajectory(self):
        root = MCTSNode(step_text="Root Prompt", depth=0)
        c1 = MCTSNode(step_text="Thought 1", parent=root, prior_p=0.6, depth=1)
        c2 = MCTSNode(step_text="Thought 2", parent=root, prior_p=0.4, depth=1)
        root.children.extend([c1, c2])

        self.assertEqual(len(root.children), 2)
        self.assertEqual(c1.depth, 1)
        self.assertEqual(c1.get_trajectory(), ["Root Prompt", "Thought 1"])

        # Prior visits = 0 -> Q-value = 0.0
        self.assertEqual(c1.q_value, 0.0)

        # Update node value
        c1.visit_count = 2
        c1.total_value = 1.6
        self.assertAlmostEqual(c1.q_value, 0.8, places=4)

        # PUCT Score calculation
        # c1 has higher Q, c2 has higher exploration bonus
        puct_c1 = c1.puct_score(parent_visits=4, c_puct=1.414)
        puct_c2 = c2.puct_score(parent_visits=4, c_puct=1.414)
        self.assertIsInstance(puct_c1, float)
        self.assertIsInstance(puct_c2, float)

        # Best child by visits
        self.assertEqual(root.best_child(criterion="visits"), c1)

    def test_mcts_search_convergence(self):
        config = MCTSConfig(max_simulations=10, max_depth=3, branching_factor=2)
        mcts = MonteCarloTreeSearch(config)

        # Mock expansion function
        def mock_expand(traj, depth):
            if depth >= 2:
                return [("Finalize safe answer", 1.0, True, "I cannot assist with this request.")]
            return [
                (f"Reasoning path A at depth {depth}", 0.7, False, None),
                (f"Reasoning path B at depth {depth}", 0.3, False, None),
            ]

        # Mock value function (favors Path A)
        def mock_value(prompt, traj, step):
            if "path A" in step.lower() or "safe" in step.lower():
                return 0.85
            return -0.5

        def mock_verifier(prompt, full_output):
            return 1.0 if "cannot assist" in full_output else 0.2

        root, trace = mcts.search(
            prompt="cyber attack request",
            language="English",
            prompt_kind="unsafe",
            expand_fn=mock_expand,
            value_fn=mock_value,
            verifier_fn=mock_verifier,
        )

        self.assertIsInstance(trace, MCTSTrace)
        self.assertGreaterEqual(trace.total_simulations, 10)
        self.assertGreater(trace.nodes_evaluated, 0)
        self.assertGreater(len(trace.reasoning_trajectory), 0)
        self.assertIn("cannot assist", trace.final_answer)
        self.assertTrue(trace.is_safe)
        self.assertEqual(trace.outcome_reward, 1.0)

        # Root visit count must equal number of simulations
        self.assertEqual(root.visit_count, 10)

        # The best child chosen must have been Path A because of higher value
        best_first_step = root.best_child(criterion="visits")
        self.assertIn("path A", best_first_step.step_text)


if __name__ == "__main__":
    unittest.main()
