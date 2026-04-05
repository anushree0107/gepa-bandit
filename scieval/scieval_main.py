#!/usr/bin/env python3
"""
SciEval + Surrogate-Guided Bandit Prompt Optimization
=====================================================

Uses the integrated GEPA components:
  - ``gepa.core.bandit_engine.SurrogateBanditEngine``
  - ``gepa.strategies.surrogate.SurrogateModel``
  - ``gepa.core.callbacks`` (including bandit-specific events)

Endpoint: local Ollama at http://10.5.30.32:11434 with qwen3:8b
Dataset:  OpenDFM/SciEval from HuggingFace
Config:   train=50, val=20, test=50, max_metric_calls=200
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
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

# Standard Python Logger Adapter for GEPA
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
    logger = logging.getLogger("scieval")
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
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.strategies.surrogate import SurrogateModel
from gepa.utils import MaxMetricCallsStopper
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://103.42.50.203:11534/api/generate"
OLLAMA_MODEL = "qwen3:8b"
TRAIN_SIZE = 50
VAL_SIZE = 30
TEST_SIZE = 50
MAX_METRIC_CALLS = 500

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "run_log.txt")


# ---------------------------------------------------------------------------
# LLM wrapper (Ollama endpoint → gepa LanguageModel protocol)
# ---------------------------------------------------------------------------

class OllamaLM:
    """Wraps the local Ollama REST endpoint as a GEPA LanguageModel."""

    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL, temperature: float = 0.7):
        self.url = url
        self.model = model
        self.temperature = temperature

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, list):
            # Convert chat messages to a single prompt string
            prompt = "\n".join(m.get("content", "") for m in prompt)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        for attempt in range(3):
            try:
                resp = requests.post(self.url, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "").strip()
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

def load_scieval_dataset(seed: int = 42):
    """Load SciEval from HuggingFace → (train, val, test) as lists of dicts."""
    rng = random.Random(seed)

    ds_valid = load_dataset("OpenDFM/SciEval", data_files="scieval-valid.json", split="train")
    valid_items = list(ds_valid)
    rng.shuffle(valid_items)

    train_set = valid_items[:TRAIN_SIZE]
    val_set = valid_items[TRAIN_SIZE: TRAIN_SIZE + VAL_SIZE]

    ds_test = load_dataset("OpenDFM/SciEval", data_files="scieval-test-local.json", split="train")
    test_items = list(ds_test)
    rng.shuffle(test_items)
    test_set = test_items[:TEST_SIZE]

    return train_set, val_set, test_set


def format_question(item: dict) -> str:
    question = item.get("question", item.get("problem", ""))
    choices = item.get("choices", item.get("options", []))
    if choices and isinstance(choices, list):
        choices_str = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
        return f"{question}\n{choices_str}"
    return question


def get_answer(item: dict) -> str:
    answer = item.get("answer", item.get("answer_text", ""))
    if isinstance(answer, list):
        return str(answer[0]) if answer else ""
    return str(answer).strip()


# ---------------------------------------------------------------------------
# SciEval Adapter (implements GEPAAdapter protocol)
# ---------------------------------------------------------------------------

class SciEvalAdapter:
    """Adapter connecting SciEval evaluation to GEPA's optimisation engine."""

    def __init__(self, lm: OllamaLM):
        self.lm = lm
        self.propose_new_texts = None  # Use default GEPA proposer

    def evaluate(
        self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        prompt = candidate.get("system_prompt", next(iter(candidate.values())))
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        for item in batch:
            question = format_question(item)
            correct = get_answer(item)

            full_prompt = f"""{prompt}

Question:
{question}

Provide ONLY the answer (the letter or the value). Do not explain."""

            prediction = self.lm(full_prompt)
            pred_clean = prediction.strip().upper()
            correct_clean = correct.strip().upper()

            letter_match = re.search(r"\b([A-D])\b", pred_clean)
            if letter_match:
                pred_clean = letter_match.group(1)

            score = 1.0 if pred_clean == correct_clean else 0.0
            feedback = f"Correct: {correct}" if score == 1.0 else f"Wrong. Predicted: {prediction.strip()}, Correct: {correct}"

            outputs.append({"prediction": prediction.strip(), "correct": correct, "score": score})
            scores.append(score)

            if capture_traces:
                trajectories.append({
                    "question": question[:200],
                    "prediction": prediction.strip(),
                    "correct": correct,
                    "score": score,
                    "feedback": feedback,
                    "prompt_used": prompt[:200],
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
                        "Inputs": {"question": trace["question"]},
                        "Generated Outputs": {"prediction": trace["prediction"]},
                        "Feedback": trace["feedback"],
                    })
            result[comp] = records
        return result


# ---------------------------------------------------------------------------
# Logging callback — prints metrics table + all prompts
# ---------------------------------------------------------------------------

class SciEvalRoundLogger:
    """Callback that logs comprehensive metrics table and all prompts."""

    def __init__(self, logger: logging.Logger, test_set: list[dict], lm: OllamaLM, max_metric_calls: int, surrogate: SurrogateModel):
        self.logger = StandardLoggerAdapter(logger)
        self.test_set = test_set
        self.lm = lm
        self.max_calls = max_metric_calls
        self.adapter = SciEvalAdapter(lm)
        self.surrogate = surrogate
        self.round_metrics: list[dict] = []
        self.best_val_score = 0.0
        self.best_test_score = 0.0
        self.best_prompt = ""
        self.last_evaluated_test_prompt = ""
        self.round_topk_scores: list[float] = []

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        self.logger.log("=" * 90)
        self.logger.log("SciEval + Surrogate-Guided Bandit — OPTIMIZATION START")
        self.logger.log(f"Config: {json.dumps(event['config'], indent=2, default=str)}")
        self.logger.log("=" * 90)

        # Capture seed candidate info so we track it from the start
        seed = event["seed_candidate"]
        self.best_prompt = next(iter(seed.values()))
        self.logger.log(f"\n  [SEED] Prompt: {self.best_prompt}")
        
        # Evaluate seed candidate on test set to establish baseline
        candidate = {"system_prompt": self.best_prompt}
        eval_result = self.adapter.evaluate(self.test_set, candidate, capture_traces=False)
        self.best_test_score = sum(eval_result.scores) / len(eval_result.scores) if eval_result.scores else 0.0
        self.last_evaluated_test_prompt = self.best_prompt
        self.logger.log(f"  [SEED] Baseline Test Accuracy: {self.best_test_score:.4f}")

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        self.round_topk_scores.append(event["average_score"])

        # Log the candidate prompt
        candidate = event["candidate"]
        prompt_text = next(iter(candidate.values()))
        status = "BEST" if event["is_best_program"] else "ACCEPTED"
        self.logger.log(
            f"\n  [{status}] Candidate #{event['candidate_idx']} "
            f"(val={event['average_score']:.4f}, parents={list(event['parent_ids'])})"
        )
        self.logger.log(f"  Prompt: {prompt_text}")

        # Always track the highest-scoring candidate
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

            # Derive best val score directly from state (authoritative source)
            best_state_idx = 0
            best_state_score = 0.0
            for i in range(len(state.program_full_scores_val_set)):
                sc = state.program_full_scores_val_set[i]
                if sc > best_state_score:
                    best_state_score = sc
                    best_state_idx = i

            # Update our tracked best if state has a higher score
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
            self.round_topk_scores = []  # Reset for next round

            # Print the full table
            self._print_table()

            self.logger.log(f"\nRound {round_num} summary: best_val={self.best_val_score:.4f}, test={test_score:.4f}, llm_calls={state.total_num_evals}")

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
        self.logger.log(f"\nBest Prompt:\n{self.best_prompt}")
        self.logger.log("=" * 90)

        # Final table
        self.logger.log("\nFINAL METRICS TABLE:")
        self._print_table()

        # All candidates
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
        file_logger.log("Loading SciEval dataset...")
        train_set, val_set, test_set = load_scieval_dataset()
        file_logger.log(f"Dataset sizes — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # LLM
        task_lm = OllamaLM(temperature=0.0)          # For answering questions
        reflection_lm = OllamaLM(temperature=0.9)    # For proposing mutations

        # Adapter
        adapter = SciEvalAdapter(task_lm)

        # Seed candidate
        seed_candidate = {
            "system_prompt": (
                "You are a science expert. Answer the following science question by selecting the correct option. "
                "Think carefully about the scientific concepts involved. "
                "Respond with ONLY the letter of the correct answer (A, B, C, or D)."
            )
        }

        # Train/val loaders
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

        # Candidate & component selectors
        candidate_selector = ParetoCandidateSelector(rng=rng)
        module_selector = RoundRobinReflectionComponentSelector()
        batch_sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=rng)

        # Callback for logging
        round_logger = SciEvalRoundLogger(base_logger, test_set, task_lm, MAX_METRIC_CALLS, surrogate)
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
            skip_perfect_score=False,
            experiment_tracker=experiment_tracker,
            reflection_lm=reflection_lm,
            callbacks=callbacks,
        )

        # The SurrogateBanditEngine
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

        file_logger.log("Starting SurrogateBanditEngine...")
        state = engine.run()

        file_logger.log(f"\nFinal state: {len(state.program_candidates)} candidates, {state.total_num_evals} total evals")

    print(f"\nDone! Log written to: {LOG_FILE}")


if __name__ == "__main__":
    main()
