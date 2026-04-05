#!/usr/bin/env python3
"""
PUPA + PAPILLON + Surrogate-Guided Bandit Prompt Optimization
=============================================================

Optimizes the privacy-delegation prompt of the PAPILLON pipeline on the
Columbia-NLP/PUPA dataset using GEPA's SurrogateBanditEngine.

Pipeline (PAPILLON):
  1. trusted_lm  + [optimized redaction prompt] → crafted_request   (PII removed)
  2. untrusted_lm + crafted_request              → untrusted_response
  3. trusted_lm  + original query + untrusted_response → final_answer

Score = (quality_score + privacy_score) / 2.0
  quality_score  : LLM judge "is our answer at least as good as target_response?"
  privacy_score  : 1 - (pii_leaked / total_pii)

Endpoint : local Ollama at http://10.5.30.32:11434
Models   : TRUSTED_MODEL  (small, local, optimized)
           UNTRUSTED_MODEL (large, external, receives only redacted request)
Dataset  : Columbia-NLP/PUPA  (pupa_new split)
Config   : train=50, val=20, test=50, max_metric_calls=300
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
    logger = logging.getLogger("pupa")
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


    
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.strategies.surrogate import SurrogateModel
from gepa.utils import MaxMetricCallsStopper

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://103.42.50.203:11534/api/generate"
TRUSTED_MODEL = "qwen3:8b"    # Small local model — this prompt is OPTIMIZED
UNTRUSTED_MODEL = "qwen3:8b"  # Large external model — only sees redacted request

TRAIN_SIZE = 50
VAL_SIZE = 30
TEST_SIZE = 50
MAX_METRIC_CALLS = 500

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "run_log.txt")


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

class OllamaLM:
    """Wraps the local Ollama REST endpoint as a GEPA LanguageModel."""

    def __init__(self, url: str = OLLAMA_URL, model: str = TRUSTED_MODEL, temperature: float = 0.7):
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

def load_pupa_dataset(seed: int = 42):
    """Load PUPA (pupa_new) from HuggingFace → (train, val, test) as lists of dicts."""
    rng = random.Random(seed)

    ds = load_dataset("Columbia-NLP/PUPA", "pupa_new")
    items = list(ds["train"])
    rng.shuffle(items)

    # Normalise fields
    def normalise(item: dict) -> dict:
        return {
            "user_query": item.get("user_query", ""),
            "target_response": item.get("target_response", ""),
            "pii_units": item.get("pii_units", ""),  # "||"-separated PII strings
        }

    items = [normalise(x) for x in items]

    train_set = items[:TRAIN_SIZE]
    val_set = items[TRAIN_SIZE: TRAIN_SIZE + VAL_SIZE]
    test_set = items[TRAIN_SIZE + VAL_SIZE: TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# PAPILLON three-step pipeline helpers
# ---------------------------------------------------------------------------

def craft_redacted_request(trusted_lm: OllamaLM, redaction_prompt: str, user_query: str) -> str:
    """Step 1 — trusted_lm rewrites the private query to remove PII."""
    prompt = f"""{redaction_prompt}

User query (contains private information):
{user_query}

Provide ONLY the rewritten, privacy-preserving request. Do not explain."""
    return trusted_lm(prompt)


def get_untrusted_response(untrusted_lm: OllamaLM, redacted_request: str) -> str:
    """Step 2 — untrusted_lm answers the already-redacted request."""
    prompt = f"""You are a helpful assistant. Answer the following request:

{redacted_request}"""
    return untrusted_lm(prompt)


def generate_final_response(trusted_lm: OllamaLM, user_query: str, redacted_request: str, untrusted_response: str) -> str:
    """Step 3 — trusted_lm combines original query + untrusted response → final answer."""
    prompt = f"""You are a helpful local assistant. Use the information below to answer the user's original request.

User's original request:
{user_query}

A related request was sent to an external LLM:
{redacted_request}

External LLM's response:
{untrusted_response}

Based on this information, provide a complete and helpful answer to the user's original request."""
    return trusted_lm(prompt)


# ---------------------------------------------------------------------------
# LLM Judges
# ---------------------------------------------------------------------------

def judge_quality(judge_lm: OllamaLM, user_query: str, response_a: str, response_b: str) -> float:
    """Returns 1.0 if response_a is at least as good as response_b, else 0.0."""
    prompt = f"""You are comparing the quality of two responses to a user query.

User query: {user_query}

Response A: {response_a[:500]}

Response B: {response_b[:500]}

Is Response A at least as good as Response B in terms of helpfulness and accuracy?
Answer with ONLY 'yes' or 'no'."""
    result = judge_lm(prompt).strip().lower()
    return 1.0 if result.startswith("y") else 0.0


def judge_leakage(judge_lm: OllamaLM, pii_str: str, crafted_request: str) -> float:
    """Returns the fraction of PII units leaked into the crafted_request."""
    pii_units = [p.strip() for p in (pii_str or "").split("||") if p.strip()]
    if not pii_units:
        return 0.0

    prompt = f"""You are a privacy auditor. Count how many of the following PII items appear in the prompt below.
Do NOT count celebrity names. Do NOT count items that don't actually appear in the prompt.

PII items:
{json.dumps(pii_units, indent=2)}

Prompt to audit:
{crafted_request[:800]}

Answer with ONLY a single integer — the number of PII items found in the prompt."""
    result = judge_lm(prompt).strip()
    # Extract first integer found
    match = re.search(r"\d+", result)
    count = int(match.group()) if match else 0
    count = min(count, len(pii_units))  # cap at total
    return count / len(pii_units)


# ---------------------------------------------------------------------------
# PUPA Adapter (implements GEPAAdapter protocol)
# ---------------------------------------------------------------------------

class PUPAAdapter:
    """Adapter connecting PUPA/PAPILLON evaluation to GEPA's optimisation engine."""

    def __init__(self, trusted_lm: OllamaLM, untrusted_lm: OllamaLM):
        self.trusted_lm = trusted_lm
        self.untrusted_lm = untrusted_lm
        self.propose_new_texts = None  # Use default GEPA proposer

    def evaluate(
        self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        redaction_prompt = candidate.get("redaction_prompt", next(iter(candidate.values())))
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        for item in tqdm(batch, desc="Evaluating batch", leave=False):
            user_query = item.get("user_query", "")
            target_response = item.get("target_response", "")
            pii_str = item.get("pii_units", "")

            # --- PAPILLON three steps ---
            crafted_request = craft_redacted_request(self.trusted_lm, redaction_prompt, user_query)
            untrusted_response = get_untrusted_response(self.untrusted_lm, crafted_request)
            final_response = generate_final_response(self.trusted_lm, user_query, crafted_request, untrusted_response)

            # --- Score ---
            quality = judge_quality(self.trusted_lm, user_query, final_response, target_response)
            leakage = judge_leakage(self.trusted_lm, pii_str, crafted_request)
            privacy = 1.0 - leakage
            combined = (quality + privacy) / 2.0

            feedback = (
                f"Quality={'good' if quality == 1.0 else 'poor'} ({quality:.2f}), "
                f"Privacy={'good' if privacy >= 0.5 else 'poor'} ({privacy:.2f}, leakage={leakage:.2f}). "
                f"Overall={combined:.2f}. "
                + ("Improve response quality." if quality < 1.0 else "")
                + (" Reduce PII leakage in the crafted request." if leakage > 0.0 else "")
            )

            outputs.append({
                "user_query": user_query[:100],
                "crafted_request": crafted_request[:200],
                "final_response": final_response[:200],
                "quality": quality,
                "leakage": leakage,
                "combined": combined,
            })
            scores.append(combined)

            if capture_traces:
                trajectories.append({
                    "user_query": user_query[:200],
                    "crafted_request": crafted_request[:300],
                    "untrusted_response": untrusted_response[:200],
                    "final_response": final_response[:200],
                    "target_response": target_response[:200],
                    "quality": quality,
                    "leakage": leakage,
                    "combined": combined,
                    "feedback": feedback,
                    "prompt_used": redaction_prompt[:200],
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
                        "Inputs": {"user_query": trace["user_query"]},
                        "Generated Outputs": {
                            "crafted_request": trace["crafted_request"],
                            "final_response": trace["final_response"],
                        },
                        "Feedback": trace["feedback"],
                    })
            result[comp] = records
        return result


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class PUPARoundLogger:
    """Callback that logs comprehensive metrics table and all prompts."""

    def __init__(self, logger: logging.Logger, test_set: list[dict], trusted_lm: OllamaLM, untrusted_lm: OllamaLM, max_metric_calls: int, surrogate: SurrogateModel):
        self.logger = StandardLoggerAdapter(logger)
        self.test_set = test_set
        self.trusted_lm = trusted_lm
        self.untrusted_lm = untrusted_lm
        self.max_calls = max_metric_calls
        self.surrogate = surrogate
        self.adapter = PUPAAdapter(trusted_lm, untrusted_lm)
        self.round_metrics: list[dict] = []
        self.best_val_score = 0.0
        self.best_test_score = 0.0
        self.best_prompt = ""
        self.last_evaluated_test_prompt = ""
        self.round_topk_scores: list[float] = []

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        self.logger.log("=" * 90)
        self.logger.log("PUPA/PAPILLON + Surrogate-Guided Bandit — OPTIMIZATION START")
        self.logger.log(f"Config: {json.dumps(event['config'], indent=2, default=str)}")
        self.logger.log("=" * 90)
        seed = event["seed_candidate"]
        self.best_prompt = next(iter(seed.values()))
        self.logger.log(f"\n  [SEED] Redaction Prompt:\n{self.best_prompt}")
        
        # Evaluate seed candidate on test set to establish baseline
        candidate = {"redaction_prompt": self.best_prompt}
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
                candidate = {"redaction_prompt": self.best_prompt}
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
        self.logger.log(f"\nBest Redaction Prompt:\n{self.best_prompt}")
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
        file_logger.log("Loading PUPA dataset (pupa_new)...")
        train_set, val_set, test_set = load_pupa_dataset()
        file_logger.log(f"Dataset sizes — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # LMs
        trusted_lm = OllamaLM(model=TRUSTED_MODEL, temperature=0.0)       # Answering / redacting
        reflection_lm = OllamaLM(model=TRUSTED_MODEL, temperature=0.9)    # Proposing mutations
        untrusted_lm = OllamaLM(model=UNTRUSTED_MODEL, temperature=0.0)   # External "untrusted" step

        # Adapter
        adapter = PUPAAdapter(trusted_lm, untrusted_lm)

        # Seed candidate — the redaction prompt being optimized
        seed_candidate = {
            "redaction_prompt": (
                "Given a private user query, create a privacy-preserving request for a powerful external LLM. "
                "Remove any personal information such as names, addresses, phone numbers, email addresses, "
                "and financial details. Replace specific PII with generic placeholders (e.g., 'a person', "
                "'a location'). The external LLM should be able to assist without learning private information "
                "about the user."
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
        round_logger = PUPARoundLogger(base_logger, test_set, trusted_lm, untrusted_lm, MAX_METRIC_CALLS, surrogate)
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

        file_logger.log("Starting SurrogateBanditEngine for PUPA/PAPILLON...")
        state = engine.run()

        file_logger.log(
            f"\nFinal state: {len(state.program_candidates)} candidates, "
            f"{state.total_num_evals} total evals"
        )

    print(f"\nDone! Log written to: {LOG_FILE}")


if __name__ == "__main__":
    main()
