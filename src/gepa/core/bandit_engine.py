"""Surrogate-guided bandit engine for GEPA optimisation.

Implements the pipeline described in ``bandit-gepa-prompt.txt``:

0. Initialisation — candidate pool, Top-P set, best prompt, surrogate, training buffer
1. Select parent prompts from Top-P
2. Generate mutations via reflective proposer
3. Score mutations cheaply with the surrogate model
4. Select Top-K candidates via UCB bandit strategy
5. Run full (expensive) LLM validation only on Top-K
6. Update candidate pool / GEPAState
7. Best-prompt update with monotonicity guarantee (val score never decreases)
8. Retrain surrogate on accumulated data
9. Repeat until budget is exhausted

This engine reuses all existing GEPA infrastructure (``GEPAState``, ``GEPAAdapter``,
``ReflectiveMutationProposer``, callbacks) and adds the surrogate/bandit layer on top.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Generic

from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.callbacks import (
    BanditSelectionEvent,
    BudgetUpdatedEvent,
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    ErrorEvent,
    GEPACallback,
    IterationEndEvent,
    IterationStartEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    ParetoFrontUpdatedEvent,
    StateSavedEvent,
    SurrogateScoredEvent,
    ValsetEvaluatedEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import (
    EvaluationCache,
    FrontierType,
    GEPAState,
    ValsetEvaluation,
    initialize_gepa_state,
)
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.proposer.reflective_mutation.reflective_mutation import (
    ReflectiveMutationProposer,
)
from gepa.strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy
from gepa.strategies.surrogate import SurrogateModel
from gepa.utils import StopperProtocol


@dataclass
class BanditConfig:
    """Hyper-parameters for the surrogate-guided bandit strategy."""

    top_p: int = 5
    """Size of the elite Top-P set (best candidates by validation score)."""

    n_parents: int = 2
    """Number of parents sampled from Top-P each round."""

    m_mutations: int = 3
    """Mutations generated per parent (total candidates = n_parents × m_mutations)."""

    top_k: int = 2
    """Number of candidates sent to full validation after surrogate scoring."""

    ucb_exploration: float = 1.0
    """UCB exploration parameter.  0 = pure exploitation."""


class SurrogateBanditEngine(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Orchestrates the surrogate-guided bandit optimisation loop.

    Generates many mutation candidates each round, filters them through a cheap
    surrogate model, and only runs expensive full LLM validation on the top-K
    survivors.  This dramatically reduces LLM calls while maintaining quality
    discovery through the UCB exploration bonus.
    """

    def __init__(
        self,
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        run_dir: str | None,
        valset: list[DataInst] | DataLoader[DataId, DataInst] | None,
        seed_candidate: dict[str, str],
        # Controls
        perfect_score: float | None,
        seed: int,
        # Proposer
        reflective_proposer: ReflectiveMutationProposer,
        frontier_type: FrontierType,
        # Bandit
        bandit_config: BanditConfig,
        surrogate: SurrogateModel | None = None,
        # Logging
        logger: LoggerProtocol | None = None,
        experiment_tracker: ExperimentTracker | None = None,
        # Callbacks
        callbacks: list[GEPACallback] | None = None,
        # Optional
        track_best_outputs: bool = False,
        raise_on_exception: bool = True,
        use_cloudpickle: bool = False,
        # Budget
        stop_callback: StopperProtocol | None = None,
        val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | None = None,
        evaluation_cache: EvaluationCache[RolloutOutput, DataId] | None = None,
    ):
        self.adapter = adapter
        self.run_dir = run_dir
        self.valset = ensure_loader(valset) if valset is not None else None
        self.seed_candidate = seed_candidate
        self.perfect_score = perfect_score
        self.seed = seed
        self.reflective_proposer = reflective_proposer
        self.frontier_type = frontier_type
        self.bandit_config = bandit_config
        self.surrogate = surrogate or SurrogateModel()
        self.logger = logger
        self.experiment_tracker = experiment_tracker
        self.callbacks = callbacks
        self.track_best_outputs = track_best_outputs
        self.raise_on_exception = raise_on_exception
        self.use_cloudpickle = use_cloudpickle
        self.stop_callback = stop_callback
        self.val_evaluation_policy: EvaluationPolicy[DataId, DataInst] = (
            val_evaluation_policy if val_evaluation_policy is not None else FullEvaluationPolicy()
        )
        self._initial_evaluation_cache = evaluation_cache
        self._stop_requested = False

        # Evaluator convenience wrapper
        def evaluator(
            batch: list[DataInst], program: dict[str, str]
        ) -> tuple[list[RolloutOutput], list[float], Sequence[dict[str, float]] | None]:
            eval_result = adapter.evaluate(batch, program, capture_traces=False)
            return eval_result.outputs, eval_result.scores, eval_result.objective_scores

        self.evaluator = evaluator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.log(msg)

    def _evaluate_on_valset(
        self,
        program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
    ) -> ValsetEvaluation[RolloutOutput, DataId]:
        valset = self.valset
        assert valset is not None
        val_ids = self.val_evaluation_policy.get_eval_batch(valset, state)
        outputs_by_id, scores_by_id, objective_by_id, num_actual_evals = state.cached_evaluate_full(
            program, list(val_ids), valset.fetch, self.evaluator
        )
        state.increment_evals(num_actual_evals)
        return ValsetEvaluation(
            outputs_by_val_id=outputs_by_id,
            scores_by_val_id=scores_by_id,
            objective_scores_by_val_id=objective_by_id,
        )

    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
        parent_program_idx: list[int],
    ) -> tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals
        valset_evaluation = self._evaluate_on_valset(new_program, state)
        state.num_full_ds_evals += 1

        new_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_evaluation=valset_evaluation,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
        )

        valset_score = self.val_evaluation_policy.get_valset_score(new_program_idx, state)
        linear_pareto_front_program_idx = self.val_evaluation_policy.get_best_program(state)
        is_best_program = new_program_idx == linear_pareto_front_program_idx

        # Snapshot Pareto front after update
        front_after = state.get_pareto_front_mapping()
        candidates_after: set[int] = set()
        for program_set in front_after.values():
            candidates_after.update(program_set)
        notify_callbacks(
            self.callbacks,
            "on_pareto_front_updated",
            ParetoFrontUpdatedEvent(
                iteration=state.i + 1,
                new_front=sorted(candidates_after),
                displaced_candidates=[],
            ),
        )

        valset = self.valset
        assert valset is not None

        notify_callbacks(
            self.callbacks,
            "on_valset_evaluated",
            ValsetEvaluatedEvent(
                iteration=state.i + 1,
                candidate_idx=new_program_idx,
                candidate=new_program,
                scores_by_val_id=dict(valset_evaluation.scores_by_val_id),
                average_score=valset_score,
                num_examples_evaluated=len(valset_evaluation.scores_by_val_id),
                total_valset_size=len(valset),
                parent_ids=parent_program_idx,
                is_best_program=is_best_program,
                outputs_by_val_id=None,
            ),
        )

        if is_best_program:
            self._log(
                f"Round {state.i + 1}: Found a better program on the valset with score {valset_score}."
            )

        if self.experiment_tracker is not None:
            log_detailed_metrics_after_discovering_new_program(
                logger=self.logger,
                gepa_state=state,
                new_program_idx=new_program_idx,
                valset_evaluation=valset_evaluation,
                objective_scores=state.prog_candidate_objective_scores[new_program_idx],
                experiment_tracker=self.experiment_tracker,
                linear_pareto_front_program_idx=linear_pareto_front_program_idx,
                valset_size=len(valset),
                val_evaluation_policy=self.val_evaluation_policy,
            )

        return new_program_idx, linear_pareto_front_program_idx

    def _should_stop(self, state: GEPAState[RolloutOutput, DataId]) -> bool:
        if self._stop_requested:
            return True
        if self.stop_callback and self.stop_callback(state):
            return True
        return False

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> GEPAState[RolloutOutput, DataId]:
        """Execute the surrogate-guided bandit optimisation loop."""
        valset = self.valset
        if valset is None:
            raise ValueError("valset must be provided to SurrogateBanditEngine.run()")

        cfg = self.bandit_config

        # ----------------------------------------------------------------
        # Notify optimisation start
        # ----------------------------------------------------------------
        notify_callbacks(
            self.callbacks,
            "on_optimization_start",
            OptimizationStartEvent(
                seed_candidate=self.seed_candidate,
                trainset_size=len(self.reflective_proposer.trainset),
                valset_size=len(valset),
                config={
                    "perfect_score": self.perfect_score,
                    "seed": self.seed,
                    "track_best_outputs": self.track_best_outputs,
                    "bandit_config": {
                        "top_p": cfg.top_p,
                        "n_parents": cfg.n_parents,
                        "m_mutations": cfg.m_mutations,
                        "top_k": cfg.top_k,
                        "ucb_exploration": cfg.ucb_exploration,
                    },
                },
            ),
        )

        # ----------------------------------------------------------------
        # Evaluate seed candidate on full valset
        # ----------------------------------------------------------------
        def valset_evaluator(
            program: dict[str, str],
        ) -> ValsetEvaluation[RolloutOutput, DataId]:
            all_ids = list(valset.all_ids())
            outputs, scores, objective_scores = self.evaluator(valset.fetch(all_ids), program)
            outputs_dict = dict(zip(all_ids, outputs, strict=False))
            scores_dict = dict(zip(all_ids, scores, strict=False))
            objective_dict = (
                dict(zip(all_ids, objective_scores, strict=False)) if objective_scores is not None else None
            )
            return ValsetEvaluation(
                outputs_by_val_id=outputs_dict,
                scores_by_val_id=scores_dict,
                objective_scores_by_val_id=objective_dict,
            )

        seed_valset_eval = valset_evaluator(self.seed_candidate)

        state = initialize_gepa_state(
            run_dir=self.run_dir,
            logger=self.logger,
            seed_candidate=self.seed_candidate,
            seed_valset_evaluation=seed_valset_eval,
            track_best_outputs=self.track_best_outputs,
            frontier_type=self.frontier_type,
            evaluation_cache=self._initial_evaluation_cache,
        )

        # Budget hook for callbacks
        def budget_hook(new_total: int, delta: int) -> None:
            notify_callbacks(
                self.callbacks,
                "on_budget_updated",
                BudgetUpdatedEvent(
                    iteration=state.i + 1,
                    metric_calls_used=new_total,
                    metric_calls_delta=delta,
                    metric_calls_remaining=None,
                ),
            )

        state.add_budget_hook(budget_hook)

        base_val_avg, _ = state.get_program_average_val_subset(0)
        self._log(f"Round 0: Seed valset score: {base_val_avg:.4f}")

        best_program_idx = 0

        # ----------------------------------------------------------------
        # Main loop — Steps 1–8
        # ----------------------------------------------------------------
        import random as stdlib_random

        rng = stdlib_random.Random(self.seed)

        while not self._should_stop(state):
            state.i += 1
            round_num = state.i + 1
            state.full_program_trace.append({"i": state.i})

            notify_callbacks(
                self.callbacks,
                "on_iteration_start",
                IterationStartEvent(
                    iteration=round_num,
                    state=state,
                    trainset_loader=self.reflective_proposer.trainset,
                ),
            )

            proposal_accepted = False

            try:
                state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)
                notify_callbacks(
                    self.callbacks,
                    "on_state_saved",
                    StateSavedEvent(iteration=round_num, run_dir=self.run_dir),
                )

                # -------------------------------------------------------
                # Step 1: SELECT PARENT PROMPTS from Top-P set
                # -------------------------------------------------------
                scores = state.program_full_scores_val_set
                sorted_indices = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
                top_p_indices = sorted_indices[: cfg.top_p]
                n_parents_actual = min(cfg.n_parents, len(top_p_indices))
                parent_indices = rng.sample(top_p_indices, n_parents_actual)

                self._log(f"Round {round_num}: Selected {n_parents_actual} parents from Top-{cfg.top_p}: {parent_indices}")

                # -------------------------------------------------------
                # Step 2: GENERATE MUTATIONS
                # -------------------------------------------------------
                mutations: list[tuple[dict[str, str], list[int]]] = []  # (candidate, parent_ids)

                for parent_idx in parent_indices:
                    for _ in range(cfg.m_mutations):
                        if self._should_stop(state):
                            break

                        # Use GEPA's reflective mutation proposer
                        proposal = self.reflective_proposer.propose(state)
                        if proposal is not None:
                            mutations.append((proposal.candidate, proposal.parent_program_ids))
                            self._log(
                                f"Round {round_num}: Generated mutation from parent {parent_idx} "
                                f"(subsample: before={sum(proposal.subsample_scores_before or []):.2f}, "
                                f"after={sum(proposal.subsample_scores_after or []):.2f})"
                            )
                        else:
                            self._log(f"Round {round_num}: Reflective proposer returned None")

                    if self._should_stop(state):
                        break

                if not mutations:
                    self._log(f"Round {round_num}: No mutations generated. Skipping round.")
                    continue

                self._log(f"Round {round_num}: Generated {len(mutations)} total mutations")

                # -------------------------------------------------------
                # Step 3: SURROGATE SCORING
                # -------------------------------------------------------
                val_questions = [str(valset.fetch([vid])[0]) for vid in valset.all_ids()]
                surrogate_scores: list[float] = []
                ucb_scores: list[float] = []
                total_pool = len(state.program_candidates) + len(mutations)

                for candidate, _ in mutations:
                    # Use the first component of the candidate as the "prompt"
                    prompt_text = next(iter(candidate.values()))
                    pred, bonus, ucb = self.surrogate.predict_with_ucb(
                        prompt_text, val_questions, cfg.ucb_exploration, total_pool
                    )
                    surrogate_scores.append(pred)
                    ucb_scores.append(ucb)

                self._log(
                    f"Round {round_num}: Surrogate scores: "
                    f"min={min(surrogate_scores):.4f}, max={max(surrogate_scores):.4f}, "
                    f"mean={sum(surrogate_scores)/len(surrogate_scores):.4f}"
                )

                # Fire surrogate scored callback
                notify_callbacks(
                    self.callbacks,
                    "on_surrogate_scored",
                    SurrogateScoredEvent(
                        iteration=round_num,
                        candidates=[m[0] for m in mutations],
                        surrogate_scores=surrogate_scores,
                        ucb_scores=ucb_scores,
                    ),
                )

                # -------------------------------------------------------
                # Step 4: SELECT TOP-K (BANDIT STRATEGY)
                # -------------------------------------------------------
                top_k_actual = min(cfg.top_k, len(mutations))
                ranked = sorted(range(len(ucb_scores)), key=lambda j: ucb_scores[j], reverse=True)
                selected_indices = ranked[:top_k_actual]
                rejected_indices = ranked[top_k_actual:]

                self._log(f"Round {round_num}: Selected Top-{top_k_actual} from {len(mutations)} by UCB: {selected_indices}")

                # Fire bandit selection callback
                notify_callbacks(
                    self.callbacks,
                    "on_bandit_selection",
                    BanditSelectionEvent(
                        iteration=round_num,
                        selected_indices=selected_indices,
                        selected_scores=[ucb_scores[i] for i in selected_indices],
                        total_candidates=len(mutations),
                    ),
                )

                # Log rejected candidates
                for idx in rejected_indices:
                    cand, pids = mutations[idx]
                    notify_callbacks(
                        self.callbacks,
                        "on_candidate_rejected",
                        CandidateRejectedEvent(
                            iteration=round_num,
                            old_score=surrogate_scores[idx],
                            new_score=ucb_scores[idx],
                            reason=f"Not in Top-{top_k_actual} by UCB (rank {ranked.index(idx)+1}/{len(mutations)})",
                        ),
                    )

                # -------------------------------------------------------
                # Step 5 & 6: FULL VALIDATION + UPDATE CANDIDATE POOL
                # -------------------------------------------------------
                topk_val_scores: list[float] = []

                for sel_idx in selected_indices:
                    if self._should_stop(state):
                        break

                    candidate, parent_ids = mutations[sel_idx]
                    new_idx, best_idx = self._run_full_eval_and_add(
                        new_program=candidate,
                        state=state,
                        parent_program_idx=parent_ids,
                    )

                    val_score = self.val_evaluation_policy.get_valset_score(new_idx, state)
                    topk_val_scores.append(val_score)
                    proposal_accepted = True

                    notify_callbacks(
                        self.callbacks,
                        "on_candidate_accepted",
                        CandidateAcceptedEvent(
                            iteration=round_num,
                            new_candidate_idx=new_idx,
                            new_score=val_score,
                            parent_ids=parent_ids,
                        ),
                    )

                    # Add data to surrogate training buffer
                    prompt_text = next(iter(candidate.values()))
                    val_subscores = state.prog_candidate_val_subscores[new_idx]
                    self.surrogate.add_data(
                        prompt_text,
                        val_questions,
                        [val_subscores.get(vid, 0.0) for vid in valset.all_ids()],
                    )

                # -------------------------------------------------------
                # Step 7: BEST PROMPT UPDATE (MONOTONICITY GUARANTEE)
                # -------------------------------------------------------
                new_best_idx = self.val_evaluation_policy.get_best_program(state)
                new_best_score = self.val_evaluation_policy.get_valset_score(new_best_idx, state)
                old_best_score = self.val_evaluation_policy.get_valset_score(best_program_idx, state)

                if new_best_score > old_best_score:
                    self._log(
                        f"Round {round_num}: ✓ IMPROVEMENT! "
                        f"{old_best_score:.4f} → {new_best_score:.4f} (candidate #{new_best_idx})"
                    )
                    best_program_idx = new_best_idx
                else:
                    self._log(
                        f"Round {round_num}: ✗ No improvement. "
                        f"Best remains at {old_best_score:.4f} (candidate #{best_program_idx})"
                    )

                # -------------------------------------------------------
                # Step 8: UPDATE SURROGATE MODEL
                # -------------------------------------------------------
                mse_loss = self.surrogate.train()
                self._log(f"Round {round_num}: Surrogate retrained. MSE loss = {mse_loss:.6f}")

                if self.experiment_tracker is not None:
                    self.experiment_tracker.log_metrics(
                        {
                            "surrogate_mse_loss": mse_loss,
                            "surrogate_buffer_size": len(self.surrogate.training_buffer),
                            "total_metric_calls": state.total_num_evals,
                            "best_val_score": self.val_evaluation_policy.get_valset_score(best_program_idx, state),
                            "num_candidates": len(state.program_candidates),
                        },
                        step=round_num,
                    )

            except Exception as e:
                self._log(f"Round {round_num}: Exception: {e}")
                self._log(traceback.format_exc())
                notify_callbacks(
                    self.callbacks,
                    "on_error",
                    ErrorEvent(
                        iteration=round_num,
                        exception=e,
                        will_continue=not self.raise_on_exception,
                    ),
                )
                if self.raise_on_exception:
                    raise
                continue
            finally:
                notify_callbacks(
                    self.callbacks,
                    "on_iteration_end",
                    IterationEndEvent(
                        iteration=round_num,
                        state=state,
                        proposal_accepted=proposal_accepted,
                    ),
                )

        # ----------------------------------------------------------------
        # Cleanup
        # ----------------------------------------------------------------
        state.save(self.run_dir, use_cloudpickle=self.use_cloudpickle)

        notify_callbacks(
            self.callbacks,
            "on_optimization_end",
            OptimizationEndEvent(
                best_candidate_idx=best_program_idx,
                total_iterations=state.i + 1,
                total_metric_calls=state.total_num_evals,
                final_state=state,
            ),
        )

        self._log(
            f"Optimisation complete. Best candidate #{best_program_idx} "
            f"with val score {self.val_evaluation_policy.get_valset_score(best_program_idx, state):.4f}"
        )

        return state

    def request_stop(self) -> None:
        """Manually request the optimisation to stop gracefully."""
        self._log("Stop requested manually. Initiating graceful shutdown...")
        self._stop_requested = True
