#!/usr/bin/env python3
"""
HoVer + Surrogate-Guided Bandit Prompt Optimization
=====================================================

Multi-hop fact verification with optimizable prompt components:
  - ``summarize_prompt``         — how to summarize retrieved evidence
  - ``create_query_hop2_prompt`` — how to generate 2nd-hop follow-up queries
  - ``create_query_hop3_prompt`` — how to generate 3rd-hop follow-up queries

Pipeline per claim (matching langProBe hover_program.py):
  Hop 1: claim → retrieve(k=7) → summarize
  Hop 2: claim + summary_1 → generate follow-up query → retrieve(k=7) → summarize
  Hop 3: claim + summary_1 + summary_2 → generate follow-up query → retrieve(k=7)
  Return: all retrieved docs across all hops

Metric: Cover Exact Match — binary 1/0 if ALL gold titles ⊆ retrieved titles
Retrieval: Snowflake Arctic Embed S (embedding-based cosine similarity)
Surrogate: Arctic Embed + regression head (drop-in replacement for TF-IDF MLP)

Endpoint: local Ollama
Dataset:  HoVer from GitHub
Config:   train=50, val=30, test=50, max_metric_calls=500
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import string
import sys
import time
import unicodedata
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
import requests

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
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils import MaxMetricCallsStopper
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer

import numpy as np


# ---------------------------------------------------------------------------
# Logger adapter (same pattern as scieval)
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
    logger = logging.getLogger("hover")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
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
TRAIN_SIZE = 50
VAL_SIZE = 30
TEST_SIZE = 50
MAX_METRIC_CALLS = 500
RETRIEVAL_K = 15  # Top-k docs per hop (matching langProBe)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "run_log.txt")

HOVER_TRAIN_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json"
HOVER_DEV_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_dev_release_v1.1.json"

WIKI_HEADERS = {
    "User-Agent": "HoVerGEPA/1.0 (research project; contact: gepa@research.edu)",
    "Accept": "application/json",
}


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
# Text normalization (matching DSPy's normalize_text for cover exact match)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for title comparison (matches DSPy's normalize_text).

    Lowercases, removes articles, removes punctuation, collapses whitespace.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


# ---------------------------------------------------------------------------
# Snowflake Arctic Embed — Embedder + Retrieval Corpus
# ---------------------------------------------------------------------------

class ArcticEmbedder:
    """Lazily loads Snowflake/snowflake-arctic-embed-s and caches embeddings."""

    def __init__(self):
        self._model = None
        self.embed_dim = 384  # snowflake-arctic-embed-s output dimension

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        print("  [Arctic] Loading Snowflake/snowflake-arctic-embed-s model ...")
        self._model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
        print(f"  [Arctic] Loaded. Embed dim = {self.embed_dim}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized embeddings. Returns ndarray [N, 384]."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings


class WikiCorpus:
    """Wikipedia corpus for BM25 retrieval.

    Fetches Wikipedia page summaries for all unique titles in the dataset PLUS
    a large set of random distractor pages, and builds a BM25 index over them.

    Uses the MediaWiki batch API (generator=random + prop=extracts) to fetch
    50 articles with their intro text per API call, making large corpora practical.
    The paper uses ColBERTv2 over ~5M Wikipedia abstracts. We simulate realistic
    difficulty by using 50K+ distractors so BM25 doesn't trivially find answers.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.titles: list[str] = []
        self.docs: list[str] = []  # "title | summary"
        self.bm25 = None

    def build(self, dataset_items: list[dict], num_distractors: int = 50000) -> None:
        """Build the corpus from dataset titles + random distractors."""
        all_titles = set()
        for item in dataset_items:
            for sf in item.get("supporting_facts", []):
                all_titles.add(sf[0])

        print(f"  [Corpus] Need {len(all_titles)} ground-truth titles + ~{num_distractors} distractors")

        corpus_cache_path = os.path.join(self.cache_dir, "wiki_corpus_large.json")

        if os.path.exists(corpus_cache_path):
            with open(corpus_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            self.titles = cached["titles"]
            self.docs = cached["docs"]
            print(f"  [Corpus] Loaded {len(self.titles)} cached docs")

            cached_set = set(self.titles)
            missing = all_titles - cached_set
            current_distract = len(self.titles) - len(all_titles)
            if not missing and current_distract >= (num_distractors * 0.8):
                self._build_bm25()
                return
            print(f"  [Corpus] {len(missing)} missing GT titles, need more distractors...")
        else:
            missing = all_titles

        # Fetch missing ground truth (one at a time — only ~200 titles)
        new_titles = []
        new_docs = []
        for i, title in enumerate(sorted(missing)):
            summary = self._fetch_wiki_summary(title)
            if summary:
                new_titles.append(title)
                new_docs.append(f"{title} | {summary}")
            if (i + 1) % 50 == 0:
                print(f"  [Corpus] Fetched {i + 1}/{len(missing)} GT titles...")

        # Fetch distractors using BATCH API (50 articles per call)
        existing_distract_count = len(self.titles) - len(all_titles)
        distractors_needed = max(0, num_distractors - existing_distract_count - len(new_titles))
        if distractors_needed > 0:
            print(f"  [Corpus] Fetching {distractors_needed} distractors via batch API...")
            exclude = all_titles | set(self.titles) | set(new_titles)
            batch_articles = self._fetch_random_articles_batch(distractors_needed, exclude)
            for title, extract in batch_articles:
                new_titles.append(title)
                new_docs.append(f"{title} | {extract}")

        self.titles.extend(new_titles)
        self.docs.extend(new_docs)

        os.makedirs(self.cache_dir, exist_ok=True)
        with open(corpus_cache_path, "w", encoding="utf-8") as f:
            json.dump({"titles": self.titles, "docs": self.docs}, f, ensure_ascii=False)
        print(f"  [Corpus] Saved cache with {len(self.titles)} docs")

        self._build_bm25()

    def _build_bm25(self):
        print(f"  [Corpus] Building BM25 index over {len(self.docs)} documents...")
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("  [Corpus] BM25 index ready.")

    def _fetch_random_articles_batch(
        self, limit: int, exclude: set[str]
    ) -> list[tuple[str, str]]:
        """Fetch random Wikipedia articles WITH extracts using batch API.

        Uses generator=random + prop=extracts to get ~50 articles per call.
        This is ~50x faster than fetching titles then summaries individually.
        """
        articles: list[tuple[str, str]] = []
        seen: set[str] = set(exclude)
        api_calls = 0
        consecutive_errors = 0

        while len(articles) < limit:
            try:
                resp = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "generator": "random",
                        "grnnamespace": 0,
                        "grnlimit": 50,
                        "prop": "extracts",
                        "exintro": True,
                        "explaintext": True,
                        "exlimit": "max",
                        "format": "json",
                    },
                    headers=WIKI_HEADERS,
                    timeout=15,
                )
                api_calls += 1
                consecutive_errors = 0

                if resp.status_code == 200:
                    pages = resp.json().get("query", {}).get("pages", {})
                    for page in pages.values():
                        title = page.get("title", "")
                        extract = page.get("extract", "")
                        if title and extract and len(extract) > 30 and title not in seen:
                            seen.add(title)
                            articles.append((title, extract))

                if api_calls % 100 == 0:
                    print(f"    ... {len(articles)}/{limit} distractors fetched ({api_calls} API calls)")

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors > 5:
                    print(f"  [Corpus] Too many errors, stopping at {len(articles)} distractors")
                    break
                import time
                time.sleep(1)

        print(f"  [Corpus] Fetched {len(articles)} distractors in {api_calls} API calls")
        return articles[:limit]

    def _fetch_wiki_summary(self, title: str) -> str:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            resp = requests.get(url, headers=WIKI_HEADERS, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("extract", "")
        except Exception:
            pass
        return ""

    def retrieve(self, query: str, k: int = 15) -> list[str]:
        if len(self.docs) == 0 or self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        return self.bm25.get_top_n(tokenized_query, self.docs, n=k)


# ---------------------------------------------------------------------------
# Arctic Embed Surrogate Model (matches SurrogateModel interface for bandit engine)
# ---------------------------------------------------------------------------

class ArcticEmbedSurrogate:
    """Snowflake Arctic Embed + regression head surrogate model.

    Drop-in replacement for gepa.strategies.surrogate.SurrogateModel,
    matching its interface: add_data(), train(), predict(), predict_with_ucb().
    Uses Snowflake/snowflake-arctic-embed-s embeddings with a learnable
    regression head instead of TF-IDF + sklearn MLP.
    """

    def __init__(self, embedder: ArcticEmbedder, min_samples: int = 5):
        import torch
        import torch.nn as nn

        self.embedder = embedder
        self.min_samples = min_samples
        self.embed_dim = embedder.embed_dim  # 384

        # Training buffer (matches SurrogateModel interface)
        self.training_buffer: list[tuple[str, float]] = []
        self.is_fitted: bool = False

        # Regression head: embedding → hidden → 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.regression_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 2, 1),
        ).to(self.device)

    def add_data(self, prompt: str, questions: list[str], scores: list[float]) -> None:
        """Append (prompt+question, score) pairs to the training buffer."""
        for q, s in zip(questions, scores):
            self.training_buffer.append((f"{prompt} [SEP] {q}", s))

    def add_data_single(self, prompt: str, question: str, score: float) -> None:
        """Append a single (prompt+question, score) to the training buffer."""
        self.training_buffer.append((f"{prompt} [SEP] {question}", score))

    def train(self) -> float:
        """(Re-)train the surrogate on all accumulated data.

        Returns MSE loss, or inf if not enough data.
        """
        import torch
        import torch.nn as nn

        if len(self.training_buffer) < self.min_samples:
            return float("inf")

        texts = [t[0] for t in self.training_buffer]
        targets = [t[1] for t in self.training_buffer]

        # Encode all texts with Arctic Embed
        embeddings_np = self.embedder.encode(texts)
        embeddings = torch.tensor(embeddings_np, dtype=torch.float32, device=self.device)
        target_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)

        # Train regression head
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(embeddings, target_tensor)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(self.regression_head.parameters(), lr=2e-5)
        loss_fn = nn.MSELoss()

        self.regression_head.train()
        avg_loss = 0.0
        for _epoch in range(3):
            total_loss = 0.0
            n_batches = 0
            for emb_batch, tgt_batch in loader:
                optimizer.zero_grad()
                preds = torch.sigmoid(self.regression_head(emb_batch).squeeze(-1))
                loss = loss_fn(preds, tgt_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)

        self.is_fitted = True
        self.regression_head.eval()
        return avg_loss

    def predict(self, prompt: str, questions: list[str]) -> list[float]:
        """Predict per-question scores for a prompt."""
        import torch

        if not self.is_fitted:
            return [0.5] * len(questions)

        texts = [f"{prompt} [SEP] {q}" for q in questions]
        embeddings_np = self.embedder.encode(texts)
        embeddings = torch.tensor(embeddings_np, dtype=torch.float32, device=self.device)

        self.regression_head.eval()
        with torch.no_grad():
            preds = torch.sigmoid(self.regression_head(embeddings).squeeze(-1))
        return [max(0.0, min(1.0, float(p))) for p in preds]

    def predict_mean(self, prompt: str, questions: list[str]) -> float:
        """Return mean predicted score."""
        preds = self.predict(prompt, questions)
        return sum(preds) / len(preds) if preds else 0.5

    def predict_with_ucb(
        self,
        prompt: str,
        questions: list[str],
        exploration: float = 1.0,
        total_candidates: int = 1,
    ) -> tuple[float, float, float]:
        """Predict score with UCB bonus (using MC Dropout for uncertainty).

        Returns: (predicted_mean, ucb_bonus, ucb_score)
        """
        import torch

        if not self.is_fitted:
            predicted = 0.5
            denom = 1 + len(self.training_buffer)
            bonus = exploration * math.sqrt(math.log(max(total_candidates, 2)) / denom)
            return predicted, bonus, predicted + bonus

        texts = [f"{prompt} [SEP] {q}" for q in questions]
        embeddings_np = self.embedder.encode(texts)
        embeddings = torch.tensor(embeddings_np, dtype=torch.float32, device=self.device)

        # MC Dropout: run multiple forward passes with dropout enabled
        self.regression_head.train()  # Enable dropout
        n_samples = 5
        all_means = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds = torch.sigmoid(self.regression_head(embeddings).squeeze(-1))
                all_means.append(preds.mean().item())

        self.regression_head.eval()

        predicted = sum(all_means) / len(all_means)
        std = (sum((m - predicted) ** 2 for m in all_means) / len(all_means)) ** 0.5

        # UCB = mean + exploration * std + exploration bonus
        denom = 1 + len(self.training_buffer)
        bonus = exploration * (std + math.sqrt(math.log(max(total_candidates, 2)) / denom))
        return predicted, bonus, predicted + bonus


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_hover_dataset(seed: int = 42):
    """Load HoVer dataset from GitHub → (train, val, test) as lists of dicts."""
    rng = random.Random(seed)

    cache_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    os.makedirs(cache_dir, exist_ok=True)

    train_path = os.path.join(cache_dir, "hover_train.json")
    dev_path = os.path.join(cache_dir, "hover_dev.json")

    for url, path in [(HOVER_TRAIN_URL, train_path), (HOVER_DEV_URL, dev_path)]:
        if not os.path.exists(path):
            print(f"Downloading {url}...")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(path, "w", encoding="utf-8") as f:
                f.write(resp.text)

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    rng.shuffle(train_data)
    rng.shuffle(dev_data)

    train_set = train_data[:TRAIN_SIZE]
    val_set = train_data[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test_set = dev_data[:TEST_SIZE]

    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# HoVer Adapter (multi-hop retrieval pipeline with Arctic Embed)
# ---------------------------------------------------------------------------

class HoVerAdapter:
    """
    Adapter connecting HoVer multi-hop fact verification to GEPA's engine.

    Uses Snowflake Arctic Embed for retrieval instead of live Wikipedia API.
    Metric: cover exact match (binary 0/1 if ALL gold titles ⊆ retrieved titles).

    Candidate dict has three prompt components:
      - summarize_prompt:         instructions for summarizing evidence
      - create_query_hop2_prompt: instructions for 2nd-hop query generation
      - create_query_hop3_prompt: instructions for 3rd-hop query generation
    """

    def __init__(self, lm: OllamaLM, corpus: WikiCorpus):
        self.lm = lm
        self.corpus = corpus
        self.propose_new_texts = None

    def _summarize_docs(self, docs: list[str], claim: str, summarize_prompt: str) -> str:
        """Summarize retrieved documents using the optimizable summarize_prompt."""
        if not docs:
            return "No relevant documents found."

        docs_text = "\n\n".join(docs[:7])  # Already in "title | summary" format
        full_prompt = f"""{summarize_prompt}

Claim: {claim}

Retrieved Documents:
{docs_text}

Summary:"""
        return self.lm(full_prompt).strip()

    def _generate_followup_query(self, claim: str, prev_summary: str, hop_prompt: str, summary_2: str = "") -> str:
        """Generate a follow-up query for the next hop."""
        if summary_2:
            full_prompt = f"""{hop_prompt}

Claim: {claim}

Summary from Hop 1:
{prev_summary}

Summary from Hop 2:
{summary_2}

Follow-up query:"""
        else:
            full_prompt = f"""{hop_prompt}

Claim: {claim}

Previous evidence summary:
{prev_summary}

Follow-up query:"""
        return self.lm(full_prompt).strip().strip('"').strip("'")

    def _run_multihop_pipeline(
        self,
        claim: str,
        candidate: dict[str, str],
    ) -> dict[str, Any]:
        """Run the full multi-hop retrieval pipeline, matching langProBe."""
        summarize_prompt = candidate.get("summarize_prompt", "")
        hop2_prompt = candidate.get("create_query_hop2_prompt", "")
        hop3_prompt = candidate.get("create_query_hop3_prompt", "")

        all_retrieved_docs: list[str] = []
        hop_details: list[dict] = []

        # HOP 1: use claim directly as query (like langProBe retrieve(claim))
        hop1_docs = self.corpus.retrieve(claim, k=RETRIEVAL_K)
        all_retrieved_docs.extend(hop1_docs)
        docs_text_1 = "\n\n".join(hop1_docs[:7])
        summary_1 = self._summarize_docs(hop1_docs, claim, summarize_prompt)
        hop_details.append({
            "query": claim,
            "docs": [d.split(" | ")[0] for d in hop1_docs],
            "summary": summary_1,
        })

        # HOP 2: generate follow-up query from claim + summary_1
        hop2_query = self._generate_followup_query(claim, summary_1, hop2_prompt)
        hop2_docs = self.corpus.retrieve(hop2_query, k=RETRIEVAL_K)
        all_retrieved_docs.extend(hop2_docs)
        docs_text_2 = "\n\n".join(hop2_docs[:7])
        summary_2 = self._summarize_docs(hop2_docs, claim, summarize_prompt)
        hop_details.append({
            "query": hop2_query,
            "docs": [d.split(" | ")[0] for d in hop2_docs],
            "summary": summary_2,
        })

        # HOP 3: generate follow-up query from claim + summary_1 + summary_2
        hop3_query = self._generate_followup_query(claim, summary_1, hop3_prompt, summary_2=summary_2)
        hop3_docs = self.corpus.retrieve(hop3_query, k=RETRIEVAL_K)
        all_retrieved_docs.extend(hop3_docs)
        hop_details.append({
            "query": hop3_query,
            "docs": [d.split(" | ")[0] for d in hop3_docs],
        })

        components_traces = {
            "summarize_prompt": [
                {"claim": claim, "passages": docs_text_1, "summary": summary_1},
                {"claim": claim, "passages": docs_text_2, "summary": summary_2},
            ],
            "create_query_hop2_prompt": [
                {"claim": claim, "summary_1": summary_1, "query": hop2_query}
            ],
            "create_query_hop3_prompt": [
                {"claim": claim, "summary_1": summary_1, "summary_2": summary_2, "query": hop3_query}
            ]
        }

        return {
            "all_retrieved_docs": all_retrieved_docs,
            "hop_details": hop_details,
            "components_traces": components_traces,
        }

    def evaluate(
        self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        for idx, item in enumerate(batch):
            claim = item["claim"]
            label = item["label"]

            # Gold titles from supporting_facts
            gt_titles = set(
                normalize_text(sf[0])
                for sf in item.get("supporting_facts", [])
            )

            result = self._run_multihop_pipeline(claim, candidate)

            # Extract titles from retrieved docs ("title | summary" format)
            found_titles = set(
                normalize_text(doc.split(" | ")[0])
                for doc in result["all_retrieved_docs"]
            )

            # Cover Exact Match: ALL gold titles must be in found titles
            cover_match = 1.0 if gt_titles.issubset(found_titles) else 0.0
            score = cover_match

            # Debug logging
            gt_raw = list(set(sf[0] for sf in item.get("supporting_facts", [])))
            retrieved_raw = list(set(doc.split(" | ")[0] for doc in result["all_retrieved_docs"]))
            print(
                f"  [EVAL {idx+1}/{len(batch)}] claim={claim[:80]}... | "
                f"cover_match={cover_match:.0f} | "
                f"GT={gt_raw} | "
                f"Retrieved={retrieved_raw[:10]}"
            )
            for h_idx, hop in enumerate(result["hop_details"]):
                print(f"    Hop {h_idx+1}: query={hop['query'][:80]} -> docs={hop['docs'][:5]}")

            feedback = (
                f"Cover exact match: {cover_match:.0f} "
                f"({len(gt_titles)} GT titles, {len(gt_titles & found_titles)} found)"
            )

            outputs.append({
                "cover_match": cover_match,
                "score": score,
                "gt_titles": list(gt_titles),
                "found_titles": list(found_titles)[:20],
                "label": label,
            })
            scores.append(score)

            if capture_traces:
                trajectories.append({
                    "claim": claim[:300],
                    "score": score,
                    "cover_match": cover_match,
                    "feedback": feedback,
                    "hop_details": result["hop_details"],
                    "components_traces": result.get("components_traces", {}),
                    "all_retrieved": list(found_titles)[:10],
                    "gold_titles": list(gt_titles),
                    "prompt_used": {k: v[:200] for k, v in candidate.items()},
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
                    feedback = trace["feedback"]
                    
                    if comp in trace.get("components_traces", {}):
                        calls = trace["components_traces"][comp]
                        for call in calls:
                            if comp == "summarize_prompt":
                                inputs = {"claim": call["claim"], "passages": call["passages"][:800]}
                                outputs = {"summary": call["summary"]}
                            elif comp == "create_query_hop2_prompt":
                                inputs = {"claim": call["claim"], "summary_1": call["summary_1"]}
                                outputs = {"query": call["query"]}
                            elif comp == "create_query_hop3_prompt":
                                inputs = {"claim": call["claim"], "summary_1": call["summary_1"], "summary_2": call["summary_2"]}
                                outputs = {"query": call["query"]}
                            else:
                                continue

                            records.append({
                                "Inputs": inputs,
                                "Generated Outputs": outputs,
                                "Feedback": feedback + f"\nGold Evidence needed: {trace['gold_titles']}. Actually retrieved overall: {trace['all_retrieved']}",
                            })
            result[comp] = records
        return result


# ---------------------------------------------------------------------------
# Logging callback — metrics table + all prompts
# ---------------------------------------------------------------------------

class HoVerRoundLogger:
    """Callback that logs comprehensive metrics table and all prompts."""

    def __init__(
        self,
        logger: logging.Logger,
        test_set: list[dict],
        lm: OllamaLM,
        max_metric_calls: int,
        surrogate: ArcticEmbedSurrogate,
        corpus: WikiCorpus,
    ):
        self.logger = StandardLoggerAdapter(logger)
        self.test_set = test_set
        self.lm = lm
        self.max_calls = max_metric_calls
        self.adapter = HoVerAdapter(lm, corpus)
        self.surrogate = surrogate
        self.round_metrics: list[dict] = []
        self.best_val_score = 0.0
        self.best_test_score = 0.0
        self.best_prompt: dict[str, str] = {}
        self.last_evaluated_test_prompt = ""
        self.round_topk_scores: list[float] = []

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        self.logger.log("=" * 90)
        self.logger.log("HoVer + Surrogate-Guided Bandit + Arctic Embed — OPTIMIZATION START")
        self.logger.log(f"Config: {json.dumps(event['config'], indent=2, default=str)}")
        self.logger.log("=" * 90)

        seed = event["seed_candidate"]
        self.best_prompt = dict(seed)
        for k, v in seed.items():
            self.logger.log(f"\n  [SEED] {k}: {v}")

        # Evaluate seed on test set for baseline
        eval_result = self.adapter.evaluate(self.test_set, seed, capture_traces=False)
        self.best_test_score = sum(eval_result.scores) / len(eval_result.scores) if eval_result.scores else 0.0
        self.last_evaluated_test_prompt = json.dumps(seed, sort_keys=True)
        self.logger.log(f"  [SEED] Baseline Test Cover Exact Match: {self.best_test_score:.4f}")

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        self.round_topk_scores.append(event["average_score"])

        candidate = event["candidate"]
        status = "BEST" if event["is_best_program"] else "ACCEPTED"
        self.logger.log(
            f"\n  [{status}] Candidate #{event['candidate_idx']} "
            f"(val={event['average_score']:.4f}, parents={list(event['parent_ids'])})"
        )
        for k, v in candidate.items():
            self.logger.log(f"    {k}: {v}")

        if event["average_score"] > self.best_val_score:
            self.best_val_score = event["average_score"]
            self.best_prompt = dict(candidate)
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
            first_key = next(iter(cand))
            self.logger.log(f"    #{i}: surrogate={s:.4f}, ucb={u:.4f} | {first_key}={cand[first_key]}")

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
            new_best_key = json.dumps(self.best_prompt, sort_keys=True)

            if new_best_key != self.last_evaluated_test_prompt:
                eval_result = self.adapter.evaluate(self.test_set, self.best_prompt, capture_traces=False)
                test_score = sum(eval_result.scores) / len(eval_result.scores) if eval_result.scores else 0.0
                self.best_test_score = test_score
                self.last_evaluated_test_prompt = new_best_key

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
                "surrogate_loss": self.surrogate.train() if self.surrogate.is_fitted else 0.0,
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
        self.logger.log(f"Best val score (cover exact match): {self.best_val_score:.4f}")
        self.logger.log(f"Best test score (cover exact match): {self.best_test_score:.4f}")
        self.logger.log("\nBest Prompts:")
        for k, v in self.best_prompt.items():
            self.logger.log(f"  {k}: {v}")
        self.logger.log("=" * 90)

        self.logger.log("\nFINAL METRICS TABLE:")
        self._print_table()

        state = event["final_state"]
        self.logger.log(f"\nALL CANDIDATES ({len(state.program_candidates)}):")
        for i, cand in enumerate(state.program_candidates):
            avg, cnt = state.get_program_average_val_subset(i)
            self.logger.log(f"  #{i}: val_avg={avg:.4f} (over {cnt} examples)")
            for k, v in cand.items():
                self.logger.log(f"    {k}: {v}")

    def _print_table(self) -> None:
        header = (
            f"{'Round':>6} | {'LLM Calls':>10} | {'Val CEM':>10} | {'Best Val CEM':>13} | "
            f"{'Test CEM':>10} | {'Surr. Loss':>11} | "
            f"{'Avg Top-k':>10} | {'Max Top-k':>10} | {'Min Top-k':>10}"
        )
        sep = "-" * len(header)
        self.logger.log(f"\n{sep}")
        self.logger.log(header)
        self.logger.log(sep)
        for m in self.round_metrics:
            loss_str = f"{m['surrogate_loss']:.6f}" if m["surrogate_loss"] != float("inf") else "     inf"
            self.logger.log(
                f"{m['rounds']:>6} | {m['llm_calls']:>10} | {m['val_accuracy']:>10.4f} | "
                f"{m['best_val_accuracy']:>13.4f} | {m['test_accuracy']:>10.4f} | {loss_str:>11} | "
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
        file_logger.log("Loading HoVer dataset...")
        train_set, val_set, test_set = load_hover_dataset()
        file_logger.log(
            f"Dataset sizes — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}"
        )

        # Initialize Arctic Embedder (shared across retrieval and surrogate)
        embedder = ArcticEmbedder()

        # Build Wikipedia corpus with distractors to simulate ColBERT but using BM25
        corpus_cache_dir = os.path.join(os.path.dirname(__file__), "data_cache")
        corpus = WikiCorpus(corpus_cache_dir)
        file_logger.log("Building Wikipedia distractor corpus to simulate ColBERT dense retrieval...")
        all_items = train_set + val_set + test_set
        corpus.build(all_items, num_distractors=50000)
        file_logger.log(f"Corpus ready: {len(corpus.docs)} documents indexed with BM25")

        # LLM
        task_lm = OllamaLM(temperature=0.0)  # For summarisation + query generation
        reflection_lm = OllamaLM(temperature=0.9)  # For proposing mutations

        # Adapter
        adapter = HoVerAdapter(task_lm, corpus)

        # Seed candidate — three prompt components
        seed_candidate = {
            "summarize_prompt": (
                "You are a helpful assistant. Summarize the following Wikipedia documents "
                "to extract facts relevant to verifying the given claim. "
                "Focus on key entities, relationships, and factual details. "
                "Be concise but include all relevant information."
            ),
            "create_query_hop2_prompt": (
                "Based on the claim and the evidence summary from the first search, "
                "generate a follow-up search query to find additional Wikipedia articles "
                "that can help verify parts of the claim not yet covered. "
                "IMPORTANT: The search engine is a strict lexical keyword matcher. "
                "Return ONLY 2-4 succinct keywords (e.g. 'Ralph Macchio Dancing Season 12'). "
                "Return ONLY the keywords, nothing else."
            ),
            "create_query_hop3_prompt": (
                "Based on the claim and two previous evidence summaries, "
                "generate a final follow-up search query to find resolving Wikipedia articles "
                "that can help verify the remaining aspects of the claim not yet covered. "
                "IMPORTANT: The search engine is a strict lexical keyword matcher. "
                "Return ONLY 2-4 succinct keywords. "
                "Return ONLY the keywords, nothing else."
            ),
        }

        # Train/val loaders
        train_loader = ListDataLoader(train_set)
        val_loader = ListDataLoader(val_set)

        # Surrogate model (Arctic Embed based — replaces TF-IDF MLP)
        surrogate = ArcticEmbedSurrogate(embedder)

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
        round_logger = HoVerRoundLogger(base_logger, test_set, task_lm, MAX_METRIC_CALLS, surrogate, corpus)
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

        file_logger.log(
            f"\nFinal state: {len(state.program_candidates)} candidates, "
            f"{state.total_num_evals} total evals"
        )

    print(f"\nDone! Log written to: {LOG_FILE}")


if __name__ == "__main__":
    main()
