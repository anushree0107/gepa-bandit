#!/usr/bin/env python3
"""
AIME + Surrogate-Guided Bandit Prompt Optimization
===================================================

Optimizes the system prompt for solving AIME math competition problems
using GEPA's SurrogateBanditEngine.

Pipeline:
  1. Load AIME problems from HuggingFace (train/val from past years, test from 2025)
  2. Optimize a "math solver" system prompt via reflective mutation + bandit selection
  3. Evaluate with exact integer match metric

Score = 1.0 if predicted integer == correct integer, else 0.0

Endpoint : local Ollama
Models   : qwen3:8b
Datasets : AI-MO/aimo-validation-aime (train/val, 90 problems)
           MathArena/aime_2025 (test, 30 problems × 5 = 150)
Config   : train=45, val=45, test=150, max_metric_calls=500
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import requests
from datasets import load_dataset

# GEPA imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.core.bandit_engine import BanditConfig, SurrogateBanditEngine
from gepa.core.callbacks import (
    BanditSelectionEvent,
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    GEPACallback,
    IterationEndEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    SurrogateScoredEvent,
    ValsetEvaluatedEvent,
)
from gepa.core.data_loader import ListDataLoader, ensure_loader
from gepa.core.state import FrontierType
from gepa.lm import LM
from gepa.logging.experiment_tracker import create_experiment_tracker
import logging
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.reflective_mutation.base import CandidateSelector, ReflectionComponentSelector
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.strategies.surrogate import SurrogateModel
from gepa.utils import MaxMetricCallsStopper


# ---------------------------------------------------------------------------
# Standard Python Logger Adapter for GEPA
# ---------------------------------------------------------------------------

class StandardLoggerAdapter(LoggerProtocol):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    def log(self, message: str):
        self.logger.info(message)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("aime")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://103.42.50.203:11534/api/generate"
OLLAMA_MODEL = "qwen3:8b"

TRAIN_SIZE = 45
VAL_SIZE = 45
TEST_REPEAT = 2  # Repeat test set 2x for statistical stability
MAX_METRIC_CALLS = 500

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "run_log.txt")


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

class OllamaLM:
    """Wraps the local Ollama REST endpoint as a GEPA LanguageModel."""

    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL, temperature: float = 0.7):
        self.url = url
        self.model = model
        self.temperature = temperature

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, list):
            prompt = "\n".join(m.get("content", "") for m in prompt)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        for attempt in range(3):
            try:
                resp = requests.post(self.url, json=payload, timeout=180)
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
            except Exception as e:
                if attempt == 2:
                    return f"[LLM_ERROR] {e}"
                time.sleep(2 ** attempt)
        return "[LLM_ERROR] max retries"

    def __repr__(self) -> str:
        return f"OllamaLM(model={self.model!r})"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_aime_dataset(seed: int = 42):
    """
    Load AIME datasets from HuggingFace.

    Train/Val: AI-MO/aimo-validation-aime (AIME 2022-2024, 90 problems)
               Shuffled and split 50/50 → 45 train, 45 val.
    Test:      MathArena/aime_2025 (AIME 2025, 30 problems × 5 repeats = 150)

    Returns (train_set, val_set, test_set) as lists of dicts with keys:
      - problem: str
      - answer: int
      - solution: str (train/val only; empty for test)
    """
    rng = random.Random(seed)

    # --- Train/Val from AIME 2022-2024 ---
    train_split = load_dataset("AI-MO/aimo-validation-aime")["train"]
    items = []
    for x in train_split:
        items.append({
            "problem": x["problem"],
            "solution": x.get("solution", ""),
            "answer": int(x["answer"]),
        })
    rng.shuffle(items)

    tot = len(items)
    train_set = items[: tot // 2]
    val_set = items[tot // 2 :]

    # --- Test from AIME 2025 ---
    test_split = load_dataset("MathArena/aime_2025")["train"]
    test_items = []
    for x in test_split:
        test_items.append({
            "problem": x["problem"],
            "answer": int(x["answer"]),
            "solution": "",  # No solutions available for 2025
        })
    test_set = test_items * TEST_REPEAT  # Repeat 5x for statistical stability

    return train_set, val_set, test_set


def save_dataset_to_csv(train_set: list, val_set: list, test_set: list, output_dir: str):
    """Save all dataset splits to CSV files for inspection."""
    os.makedirs(output_dir, exist_ok=True)

    for split_name, data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        filepath = os.path.join(output_dir, f"aime_{split_name}.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["problem", "answer", "solution"])
            writer.writeheader()
            for item in data:
                writer.writerow({
                    "problem": item["problem"],
                    "answer": item["answer"],
                    "solution": item.get("solution", ""),
                })


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_integer_answer(response: str) -> int | None:
    """
    Extract the final integer answer from the LLM response.
    AIME answers are integers from 0-999.
    """
    if not response or response.startswith("[LLM_ERROR]"):
        return None

    # Try to find a boxed answer first (common in math solutions)
    boxed_match = re.search(r"\\boxed\{(\d+)\}", response)
    if boxed_match:
        return int(boxed_match.group(1))

    # Look for "answer is X", "answer: X", "= X" at the end
    answer_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(\d+)",
        r"(?:answer|Answer|ANSWER)\s*[:=]\s*(\d+)",
        r"(\d+)\s*$",  # Last number in the response
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response.strip(), re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Fallback: find any integer in the last line
    last_line = response.strip().split("\n")[-1]
    nums = re.findall(r"\b(\d+)\b", last_line)
    if nums:
        return int(nums[-1])

    return None


# ---------------------------------------------------------------------------
# AIME Adapter (implements GEPAAdapter protocol)
# ---------------------------------------------------------------------------

class AIMEAdapter:
    """Adapter connecting AIME evaluation to GEPA's optimisation engine."""

    def __init__(self, lm: OllamaLM):
        self.lm = lm
        self.propose_new_texts = None  # Use default GEPA proposer

    def evaluate(
        self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        system_prompt = candidate.get("system_prompt", next(iter(candidate.values())))
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        for item in tqdm(batch, desc="Evaluating batch", leave=False):
            problem = item["problem"]
            correct_answer = int(item["answer"])
            solution = item.get("solution", "")

            # Build the full prompt
            full_prompt = f"""{system_prompt}

Problem:
{problem}

Provide your step-by-step reasoning, then give your final answer as a single integer."""

            # Get LLM response
            response = self.lm(full_prompt)

            # Extract and score
            predicted = extract_integer_answer(response)
            score = 1.0 if predicted is not None and predicted == correct_answer else 0.0

            # Build feedback (like DSPy tutorial's metric_with_feedback)
            if predicted is None:
                feedback = (
                    f"Could not parse an integer answer from the response. "
                    f"The correct answer is {correct_answer}."
                )
            elif score == 1.0:
                feedback = f"Correct! The answer is {correct_answer}."
            else:
                feedback = (
                    f"Incorrect. You answered {predicted}, but the correct answer is {correct_answer}."
                )

            # Include solution in feedback when available (for reflective learning)
            if solution and score < 1.0:
                feedback += f"\n\nHere's the full step-by-step solution:\n{solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers."

            outputs.append({
                "problem": problem[:150],
                "predicted": predicted,
                "correct": correct_answer,
                "response": response[:300],
                "score": score,
            })
            scores.append(score)

            if capture_traces:
                trajectories.append({
                    "problem": problem[:300],
                    "response": response[:500],
                    "predicted": predicted,
                    "correct": correct_answer,
                    "score": score,
                    "feedback": feedback,
                    "prompt_used": system_prompt[:300],
                })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        result = {}
        for comp in components_to_update:
            records = []
            if eval_batch.trajectories:
                for trace in eval_batch.trajectories:
                    records.append({
                        "Inputs": {"problem": trace["problem"]},
                        "Generated Outputs": {
                            "predicted_answer": str(trace["predicted"]),
                            "response": trace["response"][:300],
                        },
                        "Feedback": trace["feedback"],
                    })
            result[comp] = records
        return result


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class AIMERoundLogger:
    """Callback that logs comprehensive metrics table, per-sample val results, and all prompts."""

    def __init__(
        self,
        logger: logging.Logger,
        test_set: list[dict],
        val_set: list[dict],
        lm: OllamaLM,
        max_metric_calls: int,
        surrogate: SurrogateModel,
    ):
        self.logger = StandardLoggerAdapter(logger)
        self.test_set = test_set
        self.val_set = val_set
        self.lm = lm
        self.max_calls = max_metric_calls
        self.surrogate = surrogate
        self.adapter = AIMEAdapter(lm)
        self.round_metrics: list[dict] = []
        self.best_val_score = 0.0
        self.best_test_score = 0.0
        self.best_prompt = ""
        self.last_evaluated_test_prompt = ""
        self.round_topk_scores: list[float] = []

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        self.logger.log("=" * 90)
        self.logger.log("AIME + Surrogate-Guided Bandit — OPTIMIZATION START")
        self.logger.log(f"Config: {json.dumps(event['config'], indent=2, default=str)}")
        self.logger.log("=" * 90)
        seed = event["seed_candidate"]
        self.best_prompt = next(iter(seed.values()))
        self.logger.log(f"\n  [SEED] System Prompt:\n{self.best_prompt}")

        # Evaluate seed candidate on test set to establish baseline
        candidate = {"system_prompt": self.best_prompt}
        eval_result = self.adapter.evaluate(self.test_set, candidate, capture_traces=False)
        self.best_test_score = sum(eval_result.scores) / len(eval_result.scores) if eval_result.scores else 0.0
        self.last_evaluated_test_prompt = self.best_prompt
        self.logger.log(f"  [SEED] Baseline Test Accuracy: {self.best_test_score:.4f}")

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        self.round_topk_scores.append(event["average_score"])
        candidate = event["candidate"]
        prompt_text = next(iter(candidate.values()))
        status = "BEST" if event["is_best_program"] else "ACCEPTED"
        self.logger.log(
            f"\n  [{status}] Candidate #{event['candidate_idx']} "
            f"(val={event['average_score']:.4f}, parents={list(event['parent_ids'])})"
        )
        self.logger.log(f"  Prompt: {prompt_text}")

        # --- Per-sample validation logging ---
        self.logger.log(f"\n  [VAL DETAILS] Per-sample results for Candidate #{event['candidate_idx']}:")
        candidate_dict = {"system_prompt": prompt_text}
        val_eval = self.adapter.evaluate(self.val_set, candidate_dict, capture_traces=False)
        for i, (item, sc) in enumerate(zip(self.val_set, val_eval.scores)):
            result_str = "✓" if sc == 1.0 else "✗"
            pred = val_eval.outputs[i].get("predicted", "N/A")
            correct = item["answer"]
            problem_snippet = item["problem"][:80].replace("\n", " ")
            self.logger.log(f"    [{result_str}] Val #{i:02d}: pred={pred}, correct={correct} | {problem_snippet}...")
        val_acc = sum(val_eval.scores) / len(val_eval.scores) if val_eval.scores else 0.0
        self.logger.log(f"  [VAL DETAILS] Overall: {sum(val_eval.scores):.0f}/{len(val_eval.scores)} ({val_acc:.2%})")

        if event["average_score"] > self.best_val_score:
            self.best_val_score = event["average_score"]
            self.best_prompt = prompt_text
            self.logger.log(f"  >> New best! val={event['average_score']:.4f}")

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        self.logger.log(
            f"  [ACCEPTED] Candidate #{event['new_candidate_idx']} "
            f"score={event['new_score']:.4f} parents={list(event['parent_ids'])}"
        )

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        self.logger.log(f"  [REJECTED] {event['reason']}")

    def on_surrogate_scored(self, event: SurrogateScoredEvent) -> None:
        self.logger.log(f"\n  [SURROGATE] Scored {len(event['surrogate_scores'])} candidates:")
        for i, (s, u) in enumerate(zip(event["surrogate_scores"], event["ucb_scores"])):
            cand = event["candidates"][i]
            prompt = next(iter(cand.values()))[:100]
            self.logger.log(f"    #{i}: surrogate={s:.4f}, ucb={u:.4f} | {prompt}...")

    def on_bandit_selection(self, event: BanditSelectionEvent) -> None:
        self.logger.log(
            f"  [BANDIT] Selected Top-{len(event['selected_indices'])} "
            f"from {event['total_candidates']}: indices={event['selected_indices']}"
        )

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        try:
            state = event["state"]
            round_num = event["iteration"]

            # Best val score from state
            best_state_idx = 0
            best_state_score = 0.0
            for i in range(len(state.program_full_scores_val_set)):
                sc = state.program_full_scores_val_set[i]
                if sc > best_state_score:
                    best_state_score = sc
                    best_state_idx = i

            # Monotonicity test eval: evaluate on test set ONLY when val improves
            test_score = self.best_test_score
            if self.best_prompt != self.last_evaluated_test_prompt:
                candidate = {"system_prompt": self.best_prompt}
                eval_result = self.adapter.evaluate(self.test_set, candidate, capture_traces=False)
                test_score = sum(eval_result.scores) / len(eval_result.scores) if eval_result.scores else 0.0
                self.best_test_score = test_score
                self.last_evaluated_test_prompt = self.best_prompt

            val_acc = best_state_score
            topk = self.round_topk_scores if self.round_topk_scores else [0.0]
            avg_topk = sum(topk) / len(topk)
            max_topk = max(topk)
            min_topk = min(topk)

            metrics = {
                "rounds": round_num,
                "llm_calls": state.total_num_evals,
                "val_accuracy": val_acc,
                "best_val_accuracy": self.best_val_score,
                "test_accuracy": test_score,
                "surrogate_loss": self.surrogate.train() if getattr(self.surrogate, "is_fitted", False) else 0.0,
                "avg_topk": avg_topk,
                "max_topk": max_topk,
                "min_topk": min_topk,
            }
            self.round_metrics.append(metrics)
            self.round_topk_scores = []

            self._print_table()
            self.logger.log(
                f"\nRound {round_num} summary: best_val={self.best_val_score:.4f}, "
                f"test={test_score:.4f}, llm_calls={state.total_num_evals}"
            )

        except Exception as e:
            self.logger.log(f"\n[ERROR in on_iteration_end] {e}")
            import traceback
            self.logger.log(traceback.format_exc())

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        self.logger.log("\n" + "=" * 90)
        self.logger.log("OPTIMIZATION COMPLETE")
        self.logger.log(f"Total iterations: {event['total_iterations']}")
        self.logger.log(f"Total metric calls: {event['total_metric_calls']}")
        self.logger.log(f"Best candidate: #{event['best_candidate_idx']}")
        self.logger.log(f"Best val score: {self.best_val_score:.4f}")
        self.logger.log(f"Best test score: {self.best_test_score:.4f}")
        self.logger.log(f"\nBest System Prompt:\n{self.best_prompt}")
        self.logger.log("=" * 90)

        self.logger.log("\nFINAL METRICS TABLE:")
        self._print_table()

        state = event["final_state"]
        self.logger.log(f"\nALL CANDIDATES ({len(state.program_candidates)}):")
        for i, cand in enumerate(state.program_candidates):
            avg, cnt = state.get_program_average_val_subset(i)
            prompt = next(iter(cand.values()))
            self.logger.log(f"  #{i}: val_avg={avg:.4f} (over {cnt} examples)")
            self.logger.log(f"    Prompt: {prompt}")

    def _print_table(self) -> None:
        header = (
            f"{'Rounds':>6} | {'LLM Calls':>10} | {'Val Acc':>10} | {'Best Val Acc':>13} | "
            f"{'Test Acc (Selected)':>20} | {'Surrogate Loss':>15} | "
            f"{'Avg Top-k':>10} | {'Max Top-k':>10} | {'Min Top-k':>10}"
        )
        sep = "-" * len(header)
        self.logger.log(f"\n{sep}")
        self.logger.log(header)
        self.logger.log(sep)
        for m in self.round_metrics:
            loss_str = f"{m['surrogate_loss']:.6f}" if m['surrogate_loss'] != float("inf") else "     inf"
            self.logger.log(
                f"{m['rounds']:>6} | {m['llm_calls']:>10} | {m['val_accuracy']:>10.4f} | "
                f"{m['best_val_accuracy']:>13.4f} | {m['test_accuracy']:>20.4f} | {loss_str:>15} | "
                f"{m['avg_topk']:>10.4f} | {m['max_topk']:>10.4f} | {m['min_topk']:>10.4f}"
            )
        self.logger.log(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_logger = setup_logger(LOG_FILE)
    with StandardLoggerAdapter(base_logger) as file_logger:
        file_logger.log("Loading AIME dataset...")
        train_set, val_set, test_set = load_aime_dataset()
        file_logger.log(f"Dataset sizes — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # Save datasets to CSV
        save_dataset_to_csv(train_set, val_set, test_set, OUTPUT_DIR)
        file_logger.log(f"Datasets saved to CSV in {OUTPUT_DIR}")

        # LMs
        task_lm = OllamaLM(temperature=0.0)        # For solving problems
        reflection_lm = OllamaLM(temperature=0.9)   # For proposing mutations

        # Adapter
        adapter = AIMEAdapter(task_lm)

        # Seed candidate — the system prompt being optimized
        seed_candidate = {
            "system_prompt": (
                "Solve the given math problem step by step. Think carefully about the mathematical "
                "concepts involved. Show your reasoning clearly, then provide your final answer "
                "as a single integer."
            )
        }

        # Train / val loaders
        train_loader = ListDataLoader(train_set)
        val_loader = ListDataLoader(val_set)

        # Surrogate model
        surrogate = SurrogateModel()

        # Bandit config
        bandit_cfg = BanditConfig(
            top_p=5,
            n_parents=2,
            m_mutations=3,
            top_k=2,
            ucb_exploration=1.0,
        )

        # Experiment tracker
        experiment_tracker = create_experiment_tracker()

        # Stopper
        stopper = MaxMetricCallsStopper(MAX_METRIC_CALLS)

        # RNG
        rng = random.Random(42)

        # Selectors
        candidate_selector = ParetoCandidateSelector(rng=rng)
        module_selector = RoundRobinReflectionComponentSelector()
        batch_sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=rng)

        # Callback
        round_logger = AIMERoundLogger(base_logger, test_set, val_set, task_lm, MAX_METRIC_CALLS, surrogate)
        callbacks = [round_logger]

        # Reflective mutation proposer
        reflective_proposer = ReflectiveMutationProposer(
            logger=file_logger,
            trainset=train_loader,
            adapter=adapter,
            candidate_selector=candidate_selector,
            module_selector=module_selector,
            batch_sampler=batch_sampler,
            perfect_score=1.0,
            skip_perfect_score=True,
            experiment_tracker=experiment_tracker,
            reflection_lm=reflection_lm,
            callbacks=callbacks,
        )

        # SurrogateBanditEngine
        engine = SurrogateBanditEngine(
            adapter=adapter,
            run_dir=os.path.join(OUTPUT_DIR, "bandit_run"),
            valset=val_loader,
            seed_candidate=seed_candidate,
            perfect_score=1.0,
            seed=42,
            reflective_proposer=reflective_proposer,
            frontier_type="instance",
            bandit_config=bandit_cfg,
            surrogate=surrogate,
            logger=file_logger,
            experiment_tracker=experiment_tracker,
            callbacks=callbacks,
            track_best_outputs=True,
            raise_on_exception=False,
            stop_callback=stopper,
            val_evaluation_policy=FullEvaluationPolicy(),
        )

        file_logger.log("Starting SurrogateBanditEngine for AIME...")
        state = engine.run()

        file_logger.log(
            f"\nFinal state: {len(state.program_candidates)} candidates, "
            f"{state.total_num_evals} total evals"
        )

    print(f"\nDone! Log written to: {LOG_FILE}")


if __name__ == "__main__":
    main()
