"""Opt-in performance benchmarks for the hot paths (pytest-benchmark).

Excluded from the default run — execute with ``uv run pytest -m benchmark``.
Shapes are realistic: seeded float32 L2-normalized embeddings, CLAP-like
``d = 512``, libraries of 1 000 and 10 000 tracks. The point is trend, not a
number: save a baseline (``--benchmark-save``) BEFORE optimizing a hot path and
compare (``--benchmark-compare``) after — a docstring perf claim without a
measured baseline is just a claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from moodengine.cluster import outlier_scores
from moodengine.labeling import label_tracks
from moodengine.novelty import knn_distance_scores
from moodengine.search import (
    find_neighbours_harmonic,
    find_neighbours_mmr,
    find_similar,
    near_duplicate_pairs,
)

pytestmark = pytest.mark.benchmark

_D = 512  # CLAP-like embedding width
_SIZES = [1_000, 10_000]


def _library(n: int, d: int = _D, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.fixture(params=_SIZES, ids=lambda n: f"n{n}", scope="module")
def library(request):
    n = request.param
    X = _library(n)
    names = [f"track_{i:05d}.wav" for i in range(n)]
    return X, names


def test_bench_find_similar(benchmark, library):
    X, names = library
    benchmark(find_similar, 0, X, names, top_k=20)


def test_bench_find_neighbours_mmr(benchmark, library):
    X, names = library
    benchmark(find_neighbours_mmr, 0, X, names, top_k=20)


def test_bench_find_neighbours_harmonic(benchmark, library):
    X, names = library
    rng = np.random.default_rng(1)
    camelot = [f"{rng.integers(1, 13)}{'AB'[int(rng.integers(2))]}" for _ in names]
    bpm = rng.uniform(70, 180, size=len(names)).astype(np.float64)
    benchmark(
        find_neighbours_harmonic,
        0,
        X,
        names,
        top_k=20,
        camelot=camelot,
        bpm=bpm,
        harmonic_weight=0.3,
        tempo_weight=0.3,
    )


def test_bench_knn_distance_scores(benchmark, library):
    X, _ = library
    benchmark(knn_distance_scores, X, k=10)


def test_bench_near_duplicate_pairs(benchmark, library):
    X, names = library
    benchmark(near_duplicate_pairs, X, names, threshold=0.9)


def test_bench_label_tracks(benchmark, library, make_fake_embedder):
    X, _ = library
    embedder = make_fake_embedder("clap", 48_000, dim=_D)
    benchmark(label_tracks, X, embedder)


def test_bench_outlier_scores(benchmark, library):
    X, _ = library
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 8, size=X.shape[0])
    benchmark(outlier_scores, X, labels)
