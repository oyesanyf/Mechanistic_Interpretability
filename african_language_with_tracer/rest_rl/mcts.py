#!/usr/bin/env python3
"""
Monte Carlo Tree Search (VM-MCTS) for Step-Level Deliberative Reasoning.

Implements Stage 2 inference-time tree search over reasoning thoughts/steps:
- Step-level reasoning tree nodes (<thought> Step 1 -> Step 2 -> ... -> <answer>)
- Selection: PUCT (Polynomial Upper Confidence Trees) exploration/exploitation
- Expansion: Candidate reasoning step generation with prior probabilities P(s, a)
- Simulation / Evaluation: Value model (PRM) scoring + rollout outcome verification
- Backpropagation: Value sum and visit count updates along search path
- Search trace collection for Value Model training
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable, Any

logger = logging.getLogger("rest_rl.mcts")


# ---------------------------------------------------------------------------
# Node Dataclass
# ---------------------------------------------------------------------------

class MCTSNode:
    """A node in the deliberative reasoning tree representing a thought step."""

    def __init__(
        self,
        step_text: str,
        parent: Optional[MCTSNode] = None,
        prior_p: float = 1.0,
        depth: int = 0,
        is_terminal: bool = False,
        final_answer: Optional[str] = None,
    ):
        self.step_text = step_text
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.prior_p = max(1e-4, float(prior_p))
        self.depth = depth
        self.is_terminal = is_terminal
        self.final_answer = final_answer

        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.process_reward: float = 0.0
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Mean estimated value Q(s, a) = W / N."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, parent_visits: int, c_puct: float = 1.414) -> float:
        """
        Computes PUCT score:
            PUCT = Q(s, a) + c_puct * P(s, a) * sqrt(N_parent) / (1 + N_child)
        """
        exploration = c_puct * self.prior_p * (math.sqrt(parent_visits) / (1.0 + self.visit_count))
        return self.q_value + exploration

    def get_trajectory(self) -> List[str]:
        """Returns sequence of reasoning thoughts from root to this node."""
        path: List[str] = []
        curr: Optional[MCTSNode] = self
        while curr is not None:
            if curr.step_text:
                path.append(curr.step_text)
            curr = curr.parent
        return list(reversed(path))

    def best_child(self, criterion: str = "visits") -> Optional[MCTSNode]:
        """
        Selects best child based on visit count ('visits') or Q-value ('value').
        """
        if not self.children:
            return None
        if criterion == "value":
            return max(self.children, key=lambda c: c.q_value)
        # Default: highest visit count (most robust)
        return max(self.children, key=lambda c: (c.visit_count, c.q_value))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_text": self.step_text,
            "depth": self.depth,
            "visits": self.visit_count,
            "q_value": round(self.q_value, 4),
            "prior_p": round(self.prior_p, 4),
            "is_terminal": self.is_terminal,
            "final_answer": self.final_answer,
            "num_children": len(self.children),
        }


# ---------------------------------------------------------------------------
# Search Configuration & Traces
# ---------------------------------------------------------------------------

@dataclass
class MCTSConfig:
    """Hyperparameters for Monte Carlo Tree Search."""
    c_puct: float = 1.414
    max_simulations: int = 16
    max_depth: int = 3
    branching_factor: int = 3
    gamma: float = 0.99
    temperature: float = 1.0


@dataclass
class MCTSStepTrace:
    """Trace of a single reasoning step in an MCTS trajectory."""
    depth: int
    chosen_step: str
    q_value: float
    visits: int
    all_candidates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MCTSTrace:
    """Full search trace recorded during MCTS for Value Model training."""
    prompt: str
    language: str
    prompt_kind: str
    scaffold: str
    reasoning_trajectory: List[str]
    final_answer: str
    full_text: str
    best_q_value: float
    total_simulations: int
    nodes_evaluated: int
    step_traces: List[MCTSStepTrace] = field(default_factory=list)
    outcome_reward: float = 0.0
    is_safe: bool = False
    corrective_loop_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "language": self.language,
            "prompt_kind": self.prompt_kind,
            "scaffold": self.scaffold,
            "reasoning_trajectory": self.reasoning_trajectory,
            "final_answer": self.final_answer,
            "best_q_value": self.best_q_value,
            "outcome_reward": self.outcome_reward,
            "is_safe": self.is_safe,
            "nodes_evaluated": self.nodes_evaluated,
            "corrective_loop_triggered": self.corrective_loop_triggered,
            "step_traces": [asdict(st) for st in self.step_traces],
        }


# ---------------------------------------------------------------------------
# Monte Carlo Tree Search Engine
# ---------------------------------------------------------------------------

class MonteCarloTreeSearch:
    """
    Step-level VM-MCTS search engine:
    Performs deliberative thought exploration guided by a policy model and Value Model.
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config or MCTSConfig()

    def search(
        self,
        prompt: str,
        language: str,
        prompt_kind: str,
        expand_fn: Callable[[List[str], int], List[Tuple[str, float, bool, Optional[str]]]],
        value_fn: Callable[[str, List[str], str], float],
        verifier_fn: Optional[Callable[[str, str], float]] = None,
        steering_active: bool = False,
    ) -> Tuple[MCTSNode, MCTSTrace]:
        """
        Executes MCTS search from the prompt root:
        - expand_fn: takes (trajectory, depth) -> returns list of (step_text, prior_p, is_terminal, answer)
        - value_fn: takes (prompt, trajectory, step_text) -> returns PRM value V(s) in [-1.0, 1.0]
        - verifier_fn: takes (prompt, full_output) -> returns final outcome reward R in [0.0, 1.0]

        Returns (root_node, trace).
        """
        root = MCTSNode(step_text="", depth=0)
        nodes_evaluated = 0

        # Perform MCTS simulations with multi-step rollouts
        for sim in range(self.config.max_simulations):
            path: List[MCTSNode] = [root]
            curr = root

            # 1. Selection: traverse tree via PUCT score
            while curr.is_expanded and not curr.is_terminal and curr.children:
                parent_visits = max(1, curr.visit_count)
                curr = max(curr.children, key=lambda c: c.puct_score(parent_visits, self.config.c_puct))
                path.append(curr)

            # 2. Expansion: expand leaf node into candidate reasoning steps
            if not curr.is_expanded and not curr.is_terminal and curr.depth < self.config.max_depth:
                trajectory = curr.get_trajectory()
                candidates = expand_fn(trajectory, curr.depth)
                for step_txt, prior, is_term, ans in candidates:
                    child = MCTSNode(
                        step_text=step_txt,
                        parent=curr,
                        prior_p=prior,
                        depth=curr.depth + 1,
                        is_terminal=is_term,
                        final_answer=ans,
                    )
                    curr.children.append(child)
                curr.is_expanded = True

                # If children were generated, select the highest prior child to roll out
                if curr.children:
                    curr = max(curr.children, key=lambda c: c.prior_p)
                    path.append(curr)

            # 3. Rollout / Simulation: simulate trajectory down to terminal horizon
            leaf_trajectory = curr.get_trajectory()
            nodes_evaluated += 1

            prm_val = value_fn(prompt, leaf_trajectory, curr.step_text)
            curr.process_reward = prm_val

            rollout_traj = list(leaf_trajectory)
            rollout_depth = curr.depth
            rollout_ans = curr.final_answer
            is_term = curr.is_terminal
            prm_scores = [prm_val]

            # Fast rollout to maximum depth
            while not is_term and rollout_depth < self.config.max_depth:
                rollout_cands = expand_fn(rollout_traj, rollout_depth)
                if not rollout_cands:
                    break
                # Rollout policy: prioritize highest prior candidate
                best_cand = max(rollout_cands, key=lambda c: c[1])
                step_txt, prior_p, cand_is_term, cand_ans = best_cand
                rollout_traj.append(step_txt)
                rollout_depth += 1
                if cand_ans:
                    rollout_ans = cand_ans
                if cand_is_term:
                    is_term = True
                step_v = value_fn(prompt, rollout_traj, step_txt)
                prm_scores.append(step_v)

            mean_prm = sum(prm_scores) / max(1, len(prm_scores))

            # Factor in outcome verification for the simulated trajectory
            if verifier_fn is not None:
                if not rollout_ans:
                    if prompt_kind == "unsafe":
                        rollout_ans = "I cannot fulfill requests that involve harmful or unauthorized actions."
                    else:
                        rollout_ans = "I am happy to provide the helpful explanation requested."
                thoughts_block = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(rollout_traj))
                sim_full_out = f"<thought>\n{thoughts_block}\n</thought>\n<answer>\n{rollout_ans}\n</answer>"
                outcome_reward = verifier_fn(prompt, sim_full_out)
                outcome_norm = outcome_reward * 2.0 - 1.0  # normalize to [-1.0, 1.0]
                eval_value = 0.4 * mean_prm + 0.6 * outcome_norm
            else:
                eval_value = mean_prm

            # 4. Backpropagation: traverse path in reverse, updating visit counts and discounted value
            discounted_val = eval_value
            for node in reversed(path):
                node.visit_count += 1
                node.total_value += discounted_val
                discounted_val *= self.config.gamma

        # Build final trajectory from root by selecting best visited children
        chosen_trajectory: List[str] = []
        step_traces: List[MCTSStepTrace] = []
        curr = root
        final_answer = ""
        corrective_loop_triggered = False

        while not curr.is_terminal and curr.depth < self.config.max_depth:
            # If current node has not been expanded yet, expand it
            if not curr.children:
                trajectory = curr.get_trajectory()
                candidates = expand_fn(trajectory, curr.depth)
                for step_txt, prior, is_term, ans in candidates:
                    child = MCTSNode(
                        step_text=step_txt,
                        parent=curr,
                        prior_p=prior,
                        depth=curr.depth + 1,
                        is_terminal=is_term,
                        final_answer=ans,
                    )
                    curr.children.append(child)
                curr.is_expanded = True

            if not curr.children:
                break

            # Selection: on unsafe prompts, actively prefer candidates that steer toward safety/refusal
            if prompt_kind == "unsafe":
                best_c = max(
                    curr.children,
                    key=lambda c: (
                        1 if c.q_value > 0.0 else 0,
                        c.visit_count,
                        c.q_value,
                        c.prior_p,
                    )
                )
                # Check if counterfactual unsafe/compliance candidate was explored and pruned
                has_pruned_compliance = any(c.q_value < 0.0 for c in curr.children)
                if has_pruned_compliance or best_c.q_value > 0.0:
                    corrective_loop_triggered = True
            else:
                best_c = curr.best_child(criterion="visits")
                if best_c is None or best_c.visit_count == 0:
                    best_c = max(curr.children, key=lambda c: (c.q_value, c.prior_p))

            if best_c is None:
                break

            chosen_trajectory.append(best_c.step_text)

            trace_step = MCTSStepTrace(
                depth=best_c.depth,
                chosen_step=best_c.step_text,
                q_value=round(best_c.q_value, 4),
                visits=best_c.visit_count,
                all_candidates=[c.to_dict() for c in curr.children],
            )
            step_traces.append(trace_step)

            if best_c.final_answer:
                final_answer = best_c.final_answer
            if best_c.is_terminal:
                break
            curr = best_c

        if not final_answer and curr.final_answer:
            final_answer = curr.final_answer

        # Format full output
        thoughts_str = "\n".join(f"Step {idx+1}: {step}" for idx, step in enumerate(chosen_trajectory))
        if not final_answer:
            if prompt_kind == "unsafe":
                ref_starts = ["I cannot fulfill requests that involve harmful or unauthorized actions."]
                final_answer = ref_starts[0]
            else:
                final_answer = "I will proceed with the verified helpful response."
        full_text = f"<thought>\n{thoughts_str}\n</thought>\n<answer>\n{final_answer}\n</answer>"

        # Outcome score of chosen trajectory
        outcome_score = verifier_fn(prompt, full_text) if verifier_fn is not None else 1.0

        # Best Q-value along chosen trajectory
        best_q = 0.0
        if step_traces:
            best_q = max(st.q_value for st in step_traces)
        elif root.children:
            best_q = root.best_child().q_value

        trace = MCTSTrace(
            prompt=prompt,
            language=language,
            prompt_kind=prompt_kind,
            scaffold="tree_safety",
            reasoning_trajectory=chosen_trajectory,
            final_answer=final_answer,
            full_text=full_text,
            best_q_value=round(best_q, 4),
            total_simulations=self.config.max_simulations,
            nodes_evaluated=nodes_evaluated,
            step_traces=step_traces,
            outcome_reward=outcome_score,
            is_safe=(outcome_score >= 0.70),
            corrective_loop_triggered=corrective_loop_triggered,
        )

        return root, trace
