"""Unit tests for moodengine.sequence — content-conditioned SASRec/GRU4Rec next-item radio.

Training tests need torch and skip cleanly on a light install (``importorskip``); the module import
itself must stay torch-free (subprocess-pinned below). Pins: byte-identical
determinism (same sessions/X/seed → same predict_context); on an OFF-neighbour Markov chain (successors
are NOT the acoustic nearest neighbour of their predecessor, so a kNN-of-last baseline can't win) the
next-item hit@10 / nDCG@10 is strictly > random AND > kNN-of-last, for BOTH archs, measured + logged;
state_dict round-trips as numpy; an untrained model equals the kNN-of-last baseline (warm start);
save/load_sequence_model pin the canonical npz payload (schema tag + net.* weights + config.* fields),
with the save side exercised torch-free through a duck-typed stub.
"""

from __future__ import annotations

import logging
from dataclasses import fields

import numpy as np
import pytest
from assertpy import assert_that

from moodengine.sequence import (
    SequenceConfig,
    SequenceModel,
    _evaluate_context_fn,
    _hit_mrr,
    evaluate_sequence_model,
    load_sequence_model,
    save_sequence_model,
    train_sequence_model,
)


def _markov(
    seed: int,
    d: int = 16,
    k_states: int = 6,
    per_state: int = 6,
    n_sessions: int = 35,
    length: int = 8,
):
    """Random state centroids (off-neighbour) + a deterministic cycle state→(state+1). Items cluster near
    their state; sessions are random walks. The successor of a track lives in a DIFFERENT, non-adjacent
    CLAP region, so kNN-of-last (ranking by cosine to the last item) structurally cannot find it."""
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((k_states, d)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    items, state_of = [], []
    for s in range(k_states):
        for _ in range(per_state):
            v = centroids[s] + 0.15 * rng.standard_normal(d).astype(np.float32)
            items.append(v / np.linalg.norm(v))
            state_of.append(s)
    X = np.asarray(items, dtype=np.float32)
    state_of = np.asarray(state_of)
    by_state = [np.where(state_of == s)[0] for s in range(k_states)]

    def walks(n):
        out = []
        for _ in range(n):
            s = int(rng.integers(k_states))
            walk = []
            for _ in range(length):
                walk.append(int(rng.choice(by_state[s])))
                s = (s + 1) % k_states
            out.append(walk)
        return out

    return X, walks(n_sessions), walks(20)


def _rank_of(query: np.ndarray, X: np.ndarray, target: int) -> int:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    q = query / max(float(np.linalg.norm(query)), 1e-8)
    order = np.argsort(-(Xn @ q))
    return int(np.where(order == target)[0][0])


def _metrics(query_fn, X, test_sessions):
    """Mean hit@10 / nDCG@10 of the true-next item over all (context, next) pairs; query_fn(context)→vec."""
    hits, ndcgs = [], []
    for walk in test_sessions:
        for t in range(len(walk) - 1):
            r = _rank_of(query_fn(walk[: t + 1]), X, walk[t + 1])
            hits.append(1.0 if r < 10 else 0.0)
            ndcgs.append(1.0 / np.log2(r + 2) if r < 10 else 0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs))


@pytest.mark.parametrize("arch", ["sasrec", "gru4rec"])
def test_sequence_beats_random_and_knn_baselines(arch, caplog):
    pytest.importorskip("torch")
    caplog.set_level(logging.INFO)
    X, sessions, test = _markov(0)
    model = train_sequence_model(
        sessions, X, SequenceConfig.tuned(arch, seed=0)
    )  # arch-appropriate optimizer
    rng = np.random.default_rng(7)

    m_hit, m_ndcg = _metrics(lambda ctx: model.predict_context(ctx, X), X, test)
    k_hit, k_ndcg = _metrics(lambda ctx: X[ctx[-1]], X, test)  # kNN-of-last baseline
    r_hit, r_ndcg = _metrics(
        lambda ctx: rng.standard_normal(X.shape[1]).astype(np.float32), X, test
    )
    logging.getLogger("sequence_bench").info(
        "%s next-item hit@10 / nDCG@10 — model=%.3f/%.3f  kNN-of-last=%.3f/%.3f  random=%.3f/%.3f",
        arch,
        m_hit,
        m_ndcg,
        k_hit,
        k_ndcg,
        r_hit,
        r_ndcg,
    )
    assert_that(m_hit).is_greater_than(k_hit)  # learns the transition
    assert_that(m_hit).is_greater_than(r_hit)


def test_predict_context_is_deterministic():
    pytest.importorskip("torch")
    X, sessions, _ = _markov(0)
    cfg = SequenceConfig(epochs=10, seed=0)
    a = train_sequence_model(sessions, X, cfg)
    b = train_sequence_model(sessions, X, cfg)
    for s in (sessions[0][:4], sessions[3][:6]):
        assert_that(np.array_equal(a.predict_context(s, X), b.predict_context(s, X))).is_true()


def test_state_dict_roundtrip_is_numpy_and_exact():
    pytest.importorskip("torch")
    X, sessions, _ = _markov(0)
    cfg = SequenceConfig(epochs=10, seed=0)
    model = train_sequence_model(sessions, X, cfg)
    state = model.state_dict()
    assert_that(
        all(isinstance(v, np.ndarray) for v in state.values())
    ).is_true()  # pure numpy → torch-free store
    reloaded = SequenceModel.load(state, cfg)
    s = sessions[1][:5]
    np.testing.assert_array_equal(model.predict_context(s, X), reloaded.predict_context(s, X))


def test_untrained_warm_start_equals_knn_of_last():
    pytest.importorskip("torch")
    X, _, _ = _markov(0)
    model = train_sequence_model(
        [], X, SequenceConfig(seed=0)
    )  # no sessions → warm start (untrained)
    assert_that(model.trained).is_false()
    for i in (0, 5, 30):
        ctx = model.predict_context([i], X)
        cos = float(ctx @ (X[i] / np.linalg.norm(X[i])))
        assert_that(cos).is_greater_than(0.999)  # untrained context equals the item itself


def test_edge_cases():
    pytest.importorskip("torch")
    X, sessions, _ = _markov(0)
    model = train_sequence_model(sessions, X, SequenceConfig(epochs=3, seed=0))
    d = X.shape[1]
    np.testing.assert_array_equal(model.predict_context([], X), np.zeros(d, dtype=np.float32))
    np.testing.assert_array_equal(
        model.predict_context([99999], X), np.zeros(d, dtype=np.float32)
    )  # out-of-range
    with pytest.raises(ValueError, match=r"incompatible with model dim"):
        model.predict_context([0], np.zeros((5, d + 1), dtype=np.float32))  # dim mismatch


def test_hit_mrr_hand_values():
    # ranks [1, 2, 11] at k=10: rank 11 misses -> hit = 2/3; MRR = (1 + 1/2 + 1/11)/3.
    hit, mrr = _hit_mrr([1, 2, 11], k=10)

    assert_that(hit).is_close_to(2 / 3, tolerance=1e-9)
    assert_that(mrr).is_close_to((1.0 + 0.5 + 1.0 / 11.0) / 3.0, tolerance=1e-9)


def test_hit_mrr_empty_is_zero():
    assert_that(_hit_mrr([], k=10)).is_equal_to((0.0, 0.0))


def test_evaluate_context_fn_oracle_is_perfect():
    """An oracle context (the held-out target's own vector) ranks it first -> hit@1 == MRR == 1.0.
    Torch-free: exercises the ranking core through a plain numpy context function."""
    X = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    sessions = [[0, 3], [1, 5], [2, 7]]
    targets = {0: 3, 1: 5, 2: 7}  # last item of each 2-item session

    out = _evaluate_context_fn(lambda prefix: X[targets[prefix[-1]]], sessions, X, k=1)

    assert_that(out["hit_at_k"]).is_close_to(1.0, tolerance=1e-9)
    assert_that(out["mrr"]).is_close_to(1.0, tolerance=1e-9)
    assert_that(out["n_eval"]).is_equal_to(3)


def test_evaluate_context_fn_skips_degenerate_context():
    """A near-zero context vector is skipped, not scored as a miss."""
    X = np.eye(4, dtype=np.float32)

    out = _evaluate_context_fn(
        lambda prefix: np.zeros(4, dtype=np.float32), [[0, 1], [2, 3]], X, k=2
    )

    assert_that(out["n_eval"]).is_equal_to(0)
    assert_that(out["mrr"]).is_equal_to(0.0)


def test_evaluate_sequence_model_returns_metric_keys_in_range():
    pytest.importorskip("torch")
    X, sessions, test = _markov(0)
    model = train_sequence_model(sessions, X, SequenceConfig.tuned("sasrec", seed=0))

    out = evaluate_sequence_model(model, test, X, k=10)

    assert_that(set(out)).is_equal_to({"hit_at_k", "mrr", "n_eval", "k"})
    assert_that(out["hit_at_k"]).is_between(0.0, 1.0)
    assert_that(out["mrr"]).is_between(0.0, 1.0)
    assert_that(out["k"]).is_equal_to(10)


def test_train_keeps_trained_model_when_it_beats_baseline():
    """Learnable Markov structure -> trained model beats kNN-of-last on held-out -> it is kept."""
    pytest.importorskip("torch")
    X, sessions, _ = _markov(0)

    model = train_sequence_model(sessions, X, SequenceConfig.tuned("sasrec", seed=0))

    assert_that(model.trained).is_true()


def test_train_falls_back_to_warm_start_on_unlearnable_data():
    """Random sessions over random items have no transition to learn, so the trained model cannot
    beat kNN-of-last on the held-out finals and the selector returns the warm start (trained=False)."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    sessions = [[int(i) for i in rng.integers(0, 40, size=6)] for _ in range(30)]

    model = train_sequence_model(sessions, X, SequenceConfig(epochs=60, seed=0))

    assert_that(model.trained).is_false()


def test_train_returns_warm_start_when_holdout_leaves_no_training_pairs():
    """Every session has exactly 2 items, so holding out the final transition leaves nothing to
    train on -> the warm-start baseline is returned (trained=False)."""
    pytest.importorskip("torch")
    X, _, _ = _markov(0)

    model = train_sequence_model([[0, 5], [3, 8], [10, 2]], X, SequenceConfig(seed=0))

    assert_that(model.trained).is_false()


class _StubSequenceModel:
    """Duck-type of the persistence surface save_sequence_model reads: .state_dict() + .config.

    Lets the file-format tests run torch-free — the real SequenceModel cannot even be built
    without torch, but the save path only ever touches these two members.
    """

    def __init__(
        self, state: dict[str, np.ndarray], config: SequenceConfig, trained: bool = True
    ) -> None:
        self._state = state
        self.config = config
        self.trained = trained

    def state_dict(self) -> dict[str, np.ndarray]:
        return dict(self._state)


def _stub_state(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "in_proj.weight": rng.standard_normal((4, 8)).astype(np.float32),
        "out_proj.weight": rng.standard_normal((8, 4)).astype(np.float32),
        "alpha": np.asarray(1.0, dtype=np.float32),  # 0-d, like the real residual gate
    }


def _stub_config() -> SequenceConfig:
    # Non-default value for every field, so a swapped or dropped field cannot round-trip unnoticed.
    return SequenceConfig(
        arch="gru4rec",
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.05,
        max_len=17,
        epochs=9,
        lr=0.02,
        seed=42,
    )


def test_save_sequence_model_writes_schema_net_and_config_payload(tmp_path):
    state = _stub_state()
    config = _stub_config()
    path = tmp_path / "model.npz"

    returned = save_sequence_model(_StubSequenceModel(state, config), path)

    assert_that(returned).is_equal_to(path)
    with np.load(path, allow_pickle=False) as data:
        assert_that(str(data["schema"])).is_equal_to("moodengine.sequence/1")
        for name, array in state.items():
            assert_that(data[f"net.{name}"].dtype).is_equal_to(np.float32)
            np.testing.assert_array_equal(data[f"net.{name}"], array)
        for field in fields(SequenceConfig):
            stored = data[f"config.{field.name}"]
            expected = getattr(config, field.name)
            if field.type == "str":
                assert_that(stored.dtype.kind).is_equal_to("U")
                assert_that(str(stored)).is_equal_to(expected)
            elif field.type == "int":
                assert_that(stored.dtype.kind).is_equal_to("i")
                assert_that(int(stored)).is_equal_to(expected)
            else:
                assert_that(stored.dtype.kind).is_equal_to("f")
                assert_that(float(stored)).is_close_to(expected, tolerance=1e-6)


def test_save_sequence_model_bare_path_writes_exact_path(tmp_path):
    path = tmp_path / "model_state"  # no .npz suffix on purpose

    returned = save_sequence_model(_StubSequenceModel(_stub_state(), _stub_config()), path)

    assert_that(returned).is_equal_to(path)
    assert_that(
        path.exists()
    ).is_true()  # raw np.savez(str_path) would have written model_state.npz instead
    assert_that(path.with_suffix(".npz").exists()).is_false()


def test_load_sequence_model_forwards_state_and_config_to_model_load(tmp_path, monkeypatch):
    state = _stub_state()
    config = _stub_config()
    path = save_sequence_model(_StubSequenceModel(state, config), tmp_path / "model.npz")
    captured = {}
    sentinel = object()

    def fake_load(loaded_state, loaded_config):
        captured["state"] = loaded_state
        captured["config"] = loaded_config
        return sentinel

    monkeypatch.setattr(SequenceModel, "load", staticmethod(fake_load))

    result = load_sequence_model(path)

    assert_that(result).is_same_as(sentinel)
    assert_that(captured["config"]).is_equal_to(config)  # frozen dataclass ⇒ field-wise equality
    assert_that(set(captured["state"])).is_equal_to(set(state))
    for name, array in state.items():
        # Byte-identical round-trip of the very same float32 arrays — exact equality is the point.
        np.testing.assert_array_equal(captured["state"][name], array)
        assert_that(captured["state"][name].dtype).is_equal_to(np.float32)


def test_load_sequence_model_missing_file_raises_file_not_found(tmp_path):
    missing = tmp_path / "nowhere.npz"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_sequence_model(missing)


def test_load_sequence_model_wrong_schema_raises_value_error(tmp_path):
    path = tmp_path / "wrong.npz"
    with open(path, "wb") as f:
        np.savez(f, schema=np.array("moodengine.sequence/999"))

    with pytest.raises(ValueError, match="moodengine.sequence/1"):
        load_sequence_model(path)


def test_load_sequence_model_missing_config_key_raises_value_error(tmp_path):
    full = save_sequence_model(
        _StubSequenceModel(_stub_state(), _stub_config()), tmp_path / "full.npz"
    )
    with np.load(full, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files if k != "config.d_model"}
    partial = tmp_path / "partial.npz"
    with open(partial, "wb") as f:
        np.savez(f, **payload)

    with pytest.raises(ValueError, match=r"config\.d_model"):
        load_sequence_model(partial)


def test_save_load_sequence_model_roundtrip_preserves_predict_context(tmp_path):
    pytest.importorskip("torch")
    X, sessions, _ = _markov(0)
    # This test pins persistence, not learning: force a trained network to save (a 3-epoch model
    # would fail the held-out selection guard and come back as the unsaveable warm start).
    model = train_sequence_model(
        sessions, X, SequenceConfig(epochs=3, seed=0), select_on_holdout=False
    )
    path = tmp_path / "radio.npz"

    save_sequence_model(model, path)
    reloaded = load_sequence_model(path)

    assert_that(reloaded.trained).is_true()
    session = sessions[1][:5]
    np.testing.assert_allclose(
        reloaded.predict_context(session, X), model.predict_context(session, X), atol=1e-6
    )


def test_sequence_module_imports_torch_free():
    # Importing moodengine.sequence + building a SequenceConfig must NOT load torch (all torch imports are
    # lazy inside functions/methods) — this is what lets consumers' test suites stay torch-free.
    import subprocess
    import sys

    code = (
        "import sys, moodengine.sequence as s; c = s.SequenceConfig(arch='gru4rec'); assert c.d_model == 64; "
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.')]; "
        "sys.exit('torch loaded on import: '+repr(bad)) if bad else None"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert_that(r.returncode).described_as((r.stdout + r.stderr).strip()).is_equal_to(0)


def test_load_sequence_model_missing_schema_raises_value_error(tmp_path):
    # A foreign npz (e.g. an embeddings cache) must be rejected up front, not half-parsed.
    path = tmp_path / "foreign.npz"
    with open(path, "wb") as f:
        np.savez(f, embeddings=np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="no 'schema' entry"):
        load_sequence_model(path)


def test_load_sequence_model_without_net_entries_raises_value_error(tmp_path):
    # Schema-valid but weight-less (truncated / hand-built) files stay in the
    # ValueError-with-path taxonomy instead of dying on a raw KeyError later.
    path = save_sequence_model(_StubSequenceModel({}, _stub_config()), tmp_path / "empty.npz")

    with pytest.raises(ValueError, match="has no 'net"):
        load_sequence_model(path)


def test_save_sequence_model_untrained_model_raises_value_error(tmp_path):
    stub = _StubSequenceModel(_stub_state(), _stub_config(), trained=False)

    with pytest.raises(ValueError, match="untrained"):
        save_sequence_model(stub, tmp_path / "warm.npz")

    assert_that((tmp_path / "warm.npz").exists()).is_false()


def test_save_sequence_model_payload_has_exactly_the_documented_keys(tmp_path):
    # The canonical layout is a closed set: schema + net.* + config.* and NOTHING else
    # (a stray entry would make the format writer-version-dependent).
    state = _stub_state()
    path = save_sequence_model(_StubSequenceModel(state, _stub_config()), tmp_path / "m.npz")

    with np.load(path, allow_pickle=False) as data:
        expected = (
            {"schema"}
            | {f"net.{name}" for name in state}
            | {f"config.{field.name}" for field in fields(SequenceConfig)}
        )
        assert_that(sorted(data.files)).is_equal_to(sorted(expected))


def test_save_sequence_model_object_dtype_entry_raises_value_error(tmp_path):
    state = _stub_state()
    state["rogue"] = np.array([{"nested": "dict"}], dtype=object)

    with pytest.raises(ValueError, match="object dtype"):
        save_sequence_model(_StubSequenceModel(state, _stub_config()), tmp_path / "m.npz")

    assert_that((tmp_path / "m.npz").exists()).is_false()
