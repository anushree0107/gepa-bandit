#!/usr/bin/env python3
"""
LiveBench-Math + Surrogate-Guided Bandit Prompt Optimization
===================================================================

Uses the integrated GEPA components:
  - ``gepa.core.bandit_engine.SurrogateBanditEngine``
  - ``gepa.strategies.surrogate.SurrogateModel``
  - ``gepa.core.callbacks`` (including bandit-specific events)

Endpoint: local Ollama at http://103.42.50.203:11534
Model:    qwen3:8b
Dataset:  livebench/math subset from HuggingFace
Config:   train=80, val=50, test=80, max_metric_calls=500

LiveBench is a cross-domain benchmark consisting of regularly updated questions.
We use a single-step ChainOfThought as the AI system under optimization.
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
from gepa.core.data_loader import ListDataLoader
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
    logger = logging.getLogger("livebench")
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
TRAIN_SIZE = 80
VAL_SIZE = 50
TEST_SIZE = 80
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
                time.sleep(2**attempt)
        return "[LLM_ERROR] max retries"

    def __repr__(self) -> str:
        return f"OllamaLM(model={self.model!r})"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_livebench_dataset(seed: int = 0):
    """
    Load livebench/math dataset, split into equal pieces.
    """
    rng = random.Random(seed)
    ds = load_dataset('livebench/math', split='test') # The dataset relies on the 'test' split usually
    all_items = []
    
    for item in ds:
        q = item.get("turns", [""])[0] if item.get("turns") else ""
        all_items.append({
            "question_id": item.get("question_id", ""),
            "input": q,
            "target": str(item.get("ground_truth", "")),
            "task": item.get("task", ""),
        })

    rng.shuffle(all_items)
    
    n_total = len(all_items)
    print(f"[LiveBench] Total items: {n_total}")
    
    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if n_total < total_needed:
        print(f"[LiveBench] WARNING: Only {n_total} items available, need {total_needed}")
    
    train_set = all_items[:TRAIN_SIZE]
    val_set = all_items[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test_set = all_items[TRAIN_SIZE + VAL_SIZE : TRAIN_SIZE + VAL_SIZE + TEST_SIZE]
    
    return train_set, val_set, test_set


def format_question(item: dict) -> str:
    """Format LiveBench Math item into readable question string."""
    return f"[Task: {item['task']}]\n{item['input']}"


def get_answer(item: dict) -> str:
    """Return target answer."""
    return item["target"].strip()

def normalise_answer(text: str) -> str:
    text = text.strip().lower()
    return text

# ---------------------------------------------------------------------------
# LiveBench Adapter
# ---------------------------------------------------------------------------

class LiveBenchAdapter:
    """Adapter connecting LiveBench Math evaluation to GEPA's optimisation engine."""

    def __init__(self, lm: OllamaLM):
        self.lm = lm
        self.propose_new_texts = None 

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

            # Enforces single-step ChainOfThought specification in the query text.
            full_prompt = f"""{prompt}

Question:
{question}

Provide ONLY your final complete answer at the very end."""

            prediction = self.lm(full_prompt)
            pred_norm = normalise_answer(prediction)
            correct_norm = normalise_answer(correct)

            score = 1.0 if correct_norm in pred_norm else 0.0
            feedback = (
                f"Correct! ({correct})"
                if score == 1.0
                else f"Wrong. Target ground truth: {correct}"
            )

            outputs.append({
                "prediction": prediction.strip(),
                "correct": correct,
                "score": score,
            })
            scores.append(score)

            if capture_traces:
                trajectories.append({
                    "question": question,
                    "prediction": prediction.strip(),
                    "correct": correct,
                    "score": score,
                    "feedback": feedback,
                    "prompt_used": prompt,
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
# Logging callback
# ---------------------------------------------------------------------------

class LiveBenchRoundLogger:
    """Callback that logs comprehensive metrics table and all prompts."""

    def __init__(
        self,
        logger: logging.Logger,
        test_set: list[dict],
        lm: OllamaLM,
        max_metric_calls: int,
        surrogate: SurrogateModel,
    ):
        self.logger = StandardLoggerAdapter(logger)
        self.test_set = test_set
        self.lm = lm
        self.max_calls = max_metric_calls
        self.adapter = LiveBenchAdapter(lm)
        self.surrogate = surrogate
        self.round_metrics: list[dict] = []
        self.best_val_score = 0.0
        self.best_test_score = 0.0
        self.best_prompt = ""
        self.last_evaluated_test_prompt = ""
        self.round_topk_scores: list[float] = []

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        self.logger.log("=" * 90)
        self.logger.log("LiveBench-Math + Surrogate-Guided Bandit — OPTIMIZATION START")
        self.logger.log(f"Config: {json.dumps(event['config'], indent=2, default=str)}")
        self.logger.log("=" * 90)

        seed = event["seed_candidate"]
        self.best_prompt = next(iter(seed.values()))
        self.logger.log(f"\n  [SEED] Prompt: {self.best_prompt}")

        candidate = {"system_prompt": self.best_prompt}
        eval_result = self.adapter.evaluate(self.test_set, candidate, capture_traces=False)
        self.best_test_score = sum(eval_result.scores) / len(eval_result.scores) if eval_result.scores else 0.0
        self.last_evaluated_test_prompt = self.best_prompt
        self.logger.log(f"  [SEED] Baseline Test Accuracy: {self.best_test_score:.4f}")
        
        self.round_metrics.append({
            "rounds": "Seed",
            "llm_calls": len(self.test_set),
            "val_accuracy": None,
            "best_val_accuracy": 0.0,
            "test_accuracy": self.best_test_score,
            "surrogate_loss": None,
            "avg_topk": None,
            "max_topk": None,
            "min_topk": None,
        })

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

        if event["average_score"] > self.best_val_score:
            self.best_val_score = event["average_score"]
            self.best_prompt = prompt_text
            self.logger.log(f"  >> New best! val={event['average_score']:.4f}")
            if self.round_metrics and self.round_metrics[0]["rounds"] == "Seed" and self.round_metrics[0]["best_val_accuracy"] == 0.0:
                self.round_metrics[0]["best_val_accuracy"] = self.best_val_score

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        pass

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        pass

    def on_surrogate_scored(self, event: SurrogateScoredEvent) -> None:
        pass

    def on_bandit_selection(self, event: BanditSelectionEvent) -> None:
        self.logger.log(
            f"  [BANDIT] Selected Top-{len(event['selected_indices'])} "
            f"from {event['total_candidates']}: indices={event['selected_indices']}"
        )

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        try:
            state = event["state"]
            round_num = event["iteration"]

            best_state_idx = 0
            best_state_score = 0.0
            for i in range(len(state.program_full_scores_val_set)):
                sc = state.program_full_scores_val_set[i]
                if sc > best_state_score:
                    best_state_score = sc
                    best_state_idx = i

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

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        self.logger.log("\n" + "=" * 90)
        self.logger.log("OPTIMIZATION COMPLETE")
        self.logger.log(f"Best val score: {self.best_val_score:.4f}")
        self.logger.log(f"Best test score: {self.best_test_score:.4f}")
        self.logger.log(f"\nBest Prompt:\n{self.best_prompt}")
        self.logger.log("=" * 90)

        self.logger.log("\nFINAL METRICS TABLE:")
        self._print_table()

    def _print_table(self) -> None:
        header = (
            f"{'Round':<7} | {'LLM Calls':<10} | {'Training Acc':<13} | {'Validation Acc':<15} | "
            f"{'Test Acc':<9} | {'Error / Loss':<13} | {'Top-k Score (Max / Avg / Min)':<30}"
        )
        sep = "-" * len(header)
        self.logger.log(f"\n{sep}")
        self.logger.log(header)
        self.logger.log(sep)
        for m in self.round_metrics:
            is_seed = (m["rounds"] == "Seed")
            round_str = str(m["rounds"])
            llm_calls = str(m["llm_calls"])
            
            if is_seed:
                t_acc = ""
                v_acc = f"{m['best_val_accuracy']:.4f}"
                test_acc = f"{m['test_accuracy']:.4f}"
                loss_str = ""
                top_k = ""
            else:
                t_acc = f"{m['val_accuracy']:.4f}"
                v_acc = f"{m['best_val_accuracy']:.4f}"
                test_acc = f"{m['test_accuracy']:.4f}"
                loss_str = f"{m['surrogate_loss']:.4f}" if m["surrogate_loss"] != float("inf") else "inf"
                top_k = f"{m['max_topk']:.2f} / {m['avg_topk']:.2f} / {m['min_topk']:.2f}"
                
            self.logger.log(
                f"{round_str:<7} | {llm_calls:<10} | {t_acc:<13} | {v_acc:<15} | "
                f"{test_acc:<9} | {loss_str:<13} | {top_k:<30}"
            )
        self.logger.log(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_logger = setup_logger(LOG_FILE)
    with StandardLoggerAdapter(base_logger) as file_logger:
        file_logger.log("Loading LiveBench-Math dataset (all subsets)...")
        train_set, val_set, test_set = load_livebench_dataset(seed=0)
        file_logger.log(f"Dataset sizes — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # LLM
        task_lm = OllamaLM(temperature=0.0)  
        reflection_lm = OllamaLM(temperature=0.9)  

        # Adapter
        adapter = LiveBenchAdapter(task_lm)

        # "single-step ChainOfThought as the AI system under optimization" -> the 
        # initial prompt embodies this principle.
        seed_candidate = {
            "system_prompt": (
                "You are an expert mathematician. Solve the following problem by providing a "
                "clear, single-step Chain-of-Thought reasoning. Outline your complete logical steps "
                "carefully in a single flow, then produce the final exact answer clearly at the end."
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
        round_logger = LiveBenchRoundLogger(base_logger, test_set, task_lm, MAX_METRIC_CALLS, surrogate)
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

        file_logger.log("Starting SurrogateBanditEngine for LiveBench-Math...")
        state = engine.run()

        file_logger.log(
            f"\nFinal state: {len(state.program_candidates)} candidates, "
            f"{state.total_num_evals} total evals"
        )

    print(f"\nDone! Log written to: {LOG_FILE}")


if __name__ == "__main__":
    main()
