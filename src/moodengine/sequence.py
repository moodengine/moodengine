"""Session-based next-item sequence recommender over FROZEN CLAP item embeddings (opt-in).

A local, cold-start-safe "radio" model: trained on ONE user's play history, it predicts the next track
from the current session. THE key idea (mood-first, cold-start-safe): items are NOT a learned id-embedding
table (a personal library changes; a table wouldn't generalize). Each item is its FROZEN, L2-normalized
CLAP embedding ``X[idx]``; the model learns ONLY the transition dynamics and projects the session to a
CONTEXT VECTOR back in the CLAP space, so candidate ranking is delegated to the existing exact cosine
index (the displayed score is a real cosine).

Two interchangeable encoders (``config.arch``): ``sasrec`` (causal self-attention, Kang & McAuley 2018)
and ``gru4rec`` (GRU, Hidasi et al. 2016), sharing ONE training loop and loss. A learned content residual
``context_t = out_proj(encoder(in_proj(X))_t) + α·X[s_t]`` with ``out_proj`` initialized to ZERO makes an
UNTRAINED model return exactly ``X[last]`` (the nearest-CLAP-of-last baseline) — so training can only
improve on it. Loss = full-batch sampled-softmax over the sorted-unique successor pool (``logits =
cosine(context, X[cand]) / τ``), which is the exact scoring the cosine index uses at serve time.

Torch is imported LAZILY inside every function/method, so importing this module is torch-free — torch
loads only when a model is trained, loaded, or forwarded. ``state_dict()`` returns numpy (no torch types
cross the boundary), so persistence layers stay torch-free; ``save_sequence_model`` /
``load_sequence_model`` pin the one canonical on-disk ``.npz`` layout on top of that seam.
Deterministic on a single machine: ``torch.manual_seed(seed)`` + full-batch training (no data-order
RNG) ⇒ byte-identical ``predict_context``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_TEMP = (
    0.07  # training-logit temperature (CLAP cosines are narrow; sharpen). Not stored — inference
)
_EPS = (
    1e-8  # returns an L2-normalized direction ranked by cosine, so τ is irrelevant at serve time.
)


@dataclass(frozen=True)
class SequenceConfig:
    """Immutable training/model config. ``arch`` selects the encoder; the rest are small by design (a
    personal history is tiny). ``seed`` drives ``torch.manual_seed`` for byte-identical determinism."""

    arch: str = "sasrec"  # "sasrec" (causal self-attention) | "gru4rec" (GRU)
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2  # sasrec only
    dropout: float = 0.2
    max_len: int = 50  # session truncation
    epochs: int = (
        80  # full-batch; the zero-init out_proj needs enough steps to grow off the warm start
    )
    lr: float = 3e-3
    seed: int = 0

    @classmethod
    def tuned(cls, arch: str, *, seed: int = 0) -> "SequenceConfig":
        """Arch-appropriate defaults — the two encoders have genuinely different optima (as in the
        literature, not a quirk of this code). SASRec (a pre-LN transformer) learns the transition in ~80
        full-batch epochs at lr 3e-3; GRU4Rec converges only at a higher lr (~1e-2) with a SINGLE recurrent
        layer and no inter-layer dropout — at the SASRec lr it stays pinned at the warm start (kNN-of-last).
        Use this so an opt-in ``arch`` switch also switches the optimizer, instead of silently under-training."""
        if arch == "gru4rec":
            return cls(arch="gru4rec", n_layers=1, dropout=0.0, lr=1e-2, epochs=150, seed=seed)
        return cls(arch="sasrec", seed=seed)


def _build_net(config: SequenceConfig, d: int):
    """Construct the torch net (imports torch lazily). Only the encoder differs between archs; the
    in_proj (d→d_model), out_proj (d_model→d, zero-init) and α content-residual are shared."""
    import torch
    import torch.nn as nn

    class _SeqNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.arch = config.arch
            self.in_proj = nn.Linear(d, config.d_model)
            self.drop = nn.Dropout(config.dropout)
            if config.arch == "sasrec":
                self.pos_emb = nn.Embedding(config.max_len, config.d_model)
                self.layers = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            config.d_model,
                            config.n_heads,
                            dim_feedforward=4 * config.d_model,
                            dropout=config.dropout,
                            batch_first=True,
                            norm_first=True,
                            activation="relu",
                        )
                        for _ in range(config.n_layers)
                    ]
                )
                self.final_ln = nn.LayerNorm(config.d_model)
            elif config.arch == "gru4rec":
                self.gru = nn.GRU(
                    config.d_model,
                    config.d_model,
                    config.n_layers,
                    batch_first=True,
                    dropout=(config.dropout if config.n_layers > 1 else 0.0),
                )
            else:
                raise ValueError(f"unknown arch {config.arch!r}")
            self.out_proj = nn.Linear(config.d_model, d)
            nn.init.zeros_(self.out_proj.weight)  # warm-start: untrained context == the item itself
            nn.init.zeros_(self.out_proj.bias)
            self.alpha = nn.Parameter(torch.ones(()))  # content-residual gate (init 1.0)

        def context(self, x_rows):  # x_rows: (1, L, d) frozen CLAP rows of one session
            import torch

            h = self.drop(self.in_proj(x_rows))  # (1, L, d_model)
            if self.arch == "sasrec":
                length = h.size(1)
                h = h + self.pos_emb(torch.arange(length, device=h.device))[None]
                mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=h.device), 1)
                for layer in self.layers:
                    h = layer(
                        h, src_mask=mask
                    )  # True above the diagonal ⇒ position t attends only to ≤ t
                h = self.final_ln(h)
            else:
                h, _ = self.gru(h)
            # content residual: at init (out_proj=0, α=1) context == x_rows ⇒ exactly the kNN-of-last baseline.
            return (
                self.out_proj(h) + self.alpha * x_rows
            )  # (1, L, d) context in CLAP space at every position

    return _SeqNet()


def train_sequence_model(
    sessions: list[list[int]],
    X: np.ndarray,
    config: SequenceConfig = SequenceConfig(),
    *,
    select_on_holdout: bool = True,
) -> "SequenceModel":
    """Train the content-conditioned sequence model on row-index ``sessions`` over frozen CLAP ``X``
    ``(n, d)``. Full-batch (no data-order RNG); the sampled-softmax pool is the sorted-unique
    observed successors AUGMENTED with uniform negatives drawn from the whole library, so the
    training logits rank against the same candidate space the cosine index scores at serve time (a
    pool of observed successors alone never teaches the model to outrank the rest of the library).
    Deterministic: ``torch.manual_seed(config.seed)`` seeds weight init, dropout AND the negatives.

    ``select_on_holdout`` (default ``True``) makes training never ship a model worse than its own
    kNN-of-last warm start: each session's final transition is held OUT of training, the trained
    model and the warm start are scored on those held-out transitions (leave-last-out MRR, see
    :func:`evaluate_sequence_model`), and the trained model is kept only if it STRICTLY beats the
    warm start — otherwise the warm start (``trained=False``) is returned and the decision logged.
    Pass ``select_on_holdout=False`` to train on every transition and always return the trained
    network (e.g. when the caller only needs weights to persist, not a quality guarantee).

    If there are no usable sessions (all < 2 items, or < 3 once the final transition is held out)
    or ``n < 2``, returns the WARM-START model (untrained ⇒ predicts the last item's CLAP vector,
    honestly no better than the baseline). Torch confined here."""
    import torch
    import torch.nn.functional as F

    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    if X.ndim != 2:
        raise ValueError(f"X must be (n, d); got shape {X.shape}")
    n, d = X.shape
    torch.manual_seed(int(config.seed))  # the ONLY RNG source: weight init + dropout + neg draws
    net = _build_net(config, d)
    x_all = torch.from_numpy(X)  # frozen buffer, never a Parameter → no item-id table

    max_len = int(config.max_len)
    clean = [
        f[-max_len:] for s in sessions if len(f := [int(i) for i in s if 0 <= int(i) < n]) >= 2
    ]  # in-range, ≥2 items (a t→t+1 pair)
    # Hold each session's final transition out of training so the selection metric below is scored
    # on genuinely unseen data; without selection, train on every transition.
    train_sessions = [f[:-1] for f in clean if len(f) - 1 >= 2] if select_on_holdout else clean
    n_neg = min(n, 512)  # uniform negatives per step, so training ranks against the whole library

    trained = False
    if n >= 2 and train_sessions:
        # light weight_decay only: a large decay pins the zero-init out_proj/α at the warm start (kNN-of-last)
        # so training never learns the transition displacement — measured hit@10 collapses to ≈ random.
        opt = torch.optim.Adam(net.parameters(), lr=float(config.lr), weight_decay=1e-4)
        net.train()
        for _ in range(
            int(config.epochs)
        ):  # full-batch: fixed session order, no shuffle ⇒ deterministic
            opt.zero_grad()
            ctxs, tgts = [], []
            for f in train_sessions:  # one forward per session (variable length, no padding)
                idx = torch.tensor(f, dtype=torch.long)
                context = net.context(x_all.index_select(0, idx).unsqueeze(0))[0]  # (L, d)
                ctxs.append(context[:-1])  # predict positions 0..L-2 …
                tgts.append(idx[1:])  # … → the true next row
            preds = torch.cat(ctxs, 0)  # (M, d) contexts across the whole history
            targets = torch.cat(tgts, 0)  # (M,) true-next rows
            # Pool = observed successors ∪ uniform library samples, deduped so a sampled negative
            # that coincides with a target is not turned into a false negative.
            negatives = torch.randint(0, n, (n_neg,))
            pool = torch.unique(torch.cat([targets, negatives]), sorted=True)
            inv = torch.searchsorted(pool, targets)  # exact positions: targets ⊆ sorted pool
            preds_n = preds / preds.norm(dim=1, keepdim=True).clamp_min(_EPS)
            logits = (
                preds_n @ x_all.index_select(0, pool).t()
            ) / _TEMP  # (M, K) cosine/τ — the serve scoring
            F.cross_entropy(logits, inv).backward()
            opt.step()
        trained = True

    net.eval()
    model = SequenceModel(net, config, d, trained)

    if trained and select_on_holdout and clean:
        # Ship the trained model only if it strictly beats the kNN-of-last warm start on the
        # held-out final transitions; otherwise fall back to the safe baseline.
        trained_mrr = evaluate_sequence_model(model, clean, X)["mrr"]
        baseline_mrr = _evaluate_context_fn(lambda prefix: X[prefix[-1]], clean, X)["mrr"]
        if trained_mrr <= baseline_mrr:
            logger.info(
                "trained sequence model did not beat kNN-of-last on held-out transitions "
                "(MRR %.4f <= %.4f); keeping the warm-start baseline",
                trained_mrr,
                baseline_mrr,
            )
            return SequenceModel(_build_net(config, d), config, d, trained=False)

    return model


class SequenceModel:
    """A trained (or warm-start) sequence model. ``predict_context`` returns a CLAP-space unit vector; the
    caller ranks candidates by cosine (``SimilarityIndex.by_vector``). ``state_dict``/``load`` round-trip as
    numpy so persisted models never carry torch types."""

    def __init__(self, net, config: SequenceConfig, d: int, trained: bool) -> None:
        self._net = net
        self.config = config
        self._d = int(d)
        self._trained = bool(trained)

    @property
    def trained(self) -> bool:
        return self._trained

    def predict_context(self, session: list[int], X: np.ndarray) -> np.ndarray:
        """The ``(d,)`` unit context vector predicting the item AFTER ``session`` (row indices), in CLAP
        space. Out-of-range indices are filtered; an empty/all-invalid session → ``zeros(d)`` (the caller
        gates on ``ready``, nothing fabricated). Deterministic (eval + no_grad). Raises on a dim mismatch."""
        import torch

        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X.ndim != 2 or X.shape[1] != self._d:
            raise ValueError(f"X shape {X.shape} incompatible with model dim {self._d}")
        n = X.shape[0]
        rows = [int(i) for i in session if 0 <= int(i) < n][-int(self.config.max_len) :]
        if not rows:
            return np.zeros(self._d, dtype=np.float32)
        x_all = torch.from_numpy(X)
        idx = torch.tensor(rows, dtype=torch.long)
        self._net.eval()
        with torch.no_grad():
            context = self._net.context(x_all.index_select(0, idx).unsqueeze(0))[0]  # (L, d)
        vec = context[-1]  # the context predicting the item after the most recent one
        vec = vec / vec.norm().clamp_min(_EPS)
        return vec.numpy().astype(np.float32)

    def state_dict(self) -> dict:
        """Pure ``{param_name: float32 ndarray}`` (no torch tensors) → persistence stays torch-free."""
        return {
            k: v.detach().cpu().numpy().astype(np.float32, copy=True)
            for k, v in self._net.state_dict().items()
        }

    @classmethod
    def load(cls, state: dict, config: SequenceConfig) -> "SequenceModel":
        """Rebuild from a numpy ``state`` + ``config``. The CLAP dim ``d`` is read from ``out_proj.weight``;
        ``load_state_dict(strict=True)`` catches any shape/arch mismatch loudly. A persisted model is trained."""
        import torch

        d = int(np.asarray(state["out_proj.weight"]).shape[0])
        net = _build_net(config, d)
        net.load_state_dict(
            {
                k: torch.from_numpy(np.ascontiguousarray(np.asarray(v, dtype=np.float32)))
                for k, v in state.items()
            },
            strict=True,
        )
        net.eval()
        return cls(net, config, d, trained=True)


def _hit_mrr(ranks: list[int], k: int) -> tuple[float, float]:
    """``(hit@k, MRR)`` from 1-based target ranks; ``(0.0, 0.0)`` on an empty list. Pure numpy."""
    if not ranks:
        return 0.0, 0.0
    r = np.asarray(ranks, dtype=np.float64)
    return float(np.mean(r <= int(k))), float(np.mean(1.0 / r))


def _leave_last_out_pairs(sessions: list[list[int]], n: int) -> list[tuple[list[int], int]]:
    """``[(prefix_rows, held_out_last_row), …]`` for sessions with ≥2 in-range items. Pure."""
    pairs: list[tuple[list[int], int]] = []
    for s in sessions:
        f = [int(i) for i in s if 0 <= int(i) < n]
        if len(f) >= 2:
            pairs.append((f[:-1], f[-1]))
    return pairs


def _evaluate_context_fn(
    context_fn: Callable[[list[int]], np.ndarray],
    sessions: list[list[int]],
    X: np.ndarray,
    k: int = 10,
) -> dict:
    """Leave-last-out ranking metrics for a ``prefix -> (d,) vector`` context function.

    Ranks each held-out target against ALL items by cosine to ``context_fn(prefix)`` — the exact
    serve-time scoring — and returns ``{'hit_at_k', 'mrr', 'n_eval', 'k'}``. A degenerate context
    (near-zero norm, e.g. an all-invalid prefix) is skipped rather than counted as a miss. Shared
    core of :func:`evaluate_sequence_model` (trained-model context) and the kNN-of-last baseline
    used for model selection in :func:`train_sequence_model`. Pure numpy aside from ``context_fn``.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n = X.shape[0]
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), _EPS)

    ranks: list[int] = []
    for prefix, target in _leave_last_out_pairs(sessions, n):
        vec = np.asarray(context_fn(prefix), dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vec))
        if norm < _EPS:
            continue
        sims = Xn @ (vec / norm)
        ranks.append(1 + int(np.sum(sims > sims[target])))  # 1-based; ties resolve optimistically

    hit, mrr = _hit_mrr(ranks, k)
    return {"hit_at_k": hit, "mrr": mrr, "n_eval": len(ranks), "k": int(k)}


def evaluate_sequence_model(
    model: "SequenceModel", sessions: list[list[int]], X: np.ndarray, k: int = 10
) -> dict:
    """Leave-last-out hit@k / MRR of ``model`` on ``sessions`` over frozen CLAP ``X`` ``(n, d)``.

    For each session (row indices, ≥2 in-range items) the final item is held out and ranked, by
    cosine to ``model.predict_context(prefix, X)``, against ALL items — the serve-time scoring, so
    the number reflects real next-track quality rather than training loss. Returns
    ``{'hit_at_k', 'mrr', 'n_eval', 'k'}`` (``mrr`` is mean reciprocal rank over the held-out
    targets); zeros when nothing is evaluable. Uses the model (torch) but does no training and is
    deterministic. This is the metric :func:`train_sequence_model` uses to keep the warm-start
    baseline whenever the trained model fails to beat it."""
    return _evaluate_context_fn(lambda prefix: model.predict_context(prefix, X), sessions, X, k)


_SCHEMA = "moodengine.sequence/1"  # file-format tag; bump on any breaking payload change

# Rebuild casts for the 0-d config arrays in the npz payload, keyed by the field's annotation
# (postponed annotations keep it the plain string "str"/"int"/"float"). A new SequenceConfig
# field with another type must extend this map — the loud KeyError in load points here.
_CONFIG_CASTS: dict[str, Callable[[Any], Any]] = {"str": str, "int": int, "float": float}


def save_sequence_model(model: SequenceModel, path: str | Path) -> Path:
    """Persist ``model`` to ``path`` as one self-describing ``.npz`` (weights AND config together).

    ``state_dict()``/``load()`` already round-trip the network as pure numpy, but they leave every
    consumer to invent a file format carrying BOTH the weights and the :class:`SequenceConfig`
    needed to rebuild the architecture. This pins ONE canonical layout, readable without torch:
    ``"schema"`` (the format tag ``"moodengine.sequence/1"``), one ``"net.<param>"`` float32 array
    per network parameter, and one ``"config.<field>"`` 0-d array per config field (unicode for
    strings, native ints/floats otherwise). Config fields are written by iterating
    ``dataclasses.fields``, so a future field cannot be silently dropped from the payload.

    Writes through an open file handle because ``np.savez`` appends ``.npz`` to bare paths — the
    file lands at EXACTLY ``path``. The parent directory must already exist (directory layout is
    the caller's concern). Needs no torch beyond what ``model.state_dict()`` already loaded.
    Refuses an untrained (warm-start) model with ``ValueError``: :meth:`SequenceModel.load`
    reports every loaded model as ``trained=True``, so persisting the warm-start baseline would
    hand it back mislabeled. Returns ``Path(path)``.
    """
    if not model.trained:
        raise ValueError(
            "refusing to persist an untrained (warm-start) model: a loaded model always "
            "reports trained=True, so the untrained baseline would come back mislabeled; "
            "train it first (train_sequence_model)"
        )

    # Values are all ndarrays; typed Any because np.savez's stub checks ** values against its
    # allow_pickle keyword too, and our dotted keys can never actually bind to it.
    payload: dict[str, Any] = {"schema": np.array(_SCHEMA)}

    for name, array in model.state_dict().items():
        payload[f"net.{name}"] = array

    for field in fields(SequenceConfig):
        payload[f"config.{field.name}"] = np.array(getattr(model.config, field.name))

    # Fail at write time on anything that would pickle. np.savez only grew its allow_pickle
    # keyword in numpy 2.1 — on older supported versions it would fall into **kwds and be
    # written into the archive as a bogus 'allow_pickle' entry — so the data-only contract is
    # enforced by an explicit dtype check instead; the load side refuses pickle regardless.
    bad = [key for key, value in payload.items() if np.asarray(value).dtype == object]
    if bad:
        raise ValueError(
            f"state entries {bad} have object dtype; the format is plain numeric/string "
            "arrays only (an object array would be pickled into the file and only rejected "
            "at load)"
        )

    out = Path(path)
    with open(out, "wb") as f:
        np.savez(f, **payload)
    return out


def load_sequence_model(path: str | Path) -> SequenceModel:
    """Rebuild a :class:`SequenceModel` from a file written by :func:`save_sequence_model`.

    Validates the ``"schema"`` tag, reconstructs the :class:`SequenceConfig` from the
    ``"config.<field>"`` entries (a missing field raises ``ValueError`` naming it), then delegates
    to :meth:`SequenceModel.load` — which imports torch to rebuild the network, so this function
    needs torch installed even though the file itself is plain numpy. As with
    :meth:`SequenceModel.load`, the returned model reports ``trained=True`` — safe because
    :func:`save_sequence_model` refuses to persist an untrained model in the first place.

    Raises ``FileNotFoundError`` if ``path`` does not exist, ``ValueError`` on a missing or
    unsupported schema tag, a missing ``config.<field>`` entry, or a payload without any
    ``net.<param>`` entry. Arrays are read with ``allow_pickle=False`` — the payload is
    data-only by contract, never pickled objects.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"sequence model file not found: {path}")

    with np.load(p, allow_pickle=False) as data:
        if "schema" not in data:
            raise ValueError(
                f"{path} has no 'schema' entry; expected {_SCHEMA!r} — not a sequence-model file"
            )
        schema = str(data["schema"])
        if schema != _SCHEMA:
            raise ValueError(
                f"unsupported sequence-model schema {schema!r} in {path}; expected {_SCHEMA!r}"
            )

        kwargs: dict[str, Any] = {}
        for field in fields(SequenceConfig):
            key = f"config.{field.name}"
            if key not in data:
                raise ValueError(
                    f"{path} is missing {key!r}; expected one 'config.<field>' entry "
                    f"per SequenceConfig field"
                )
            kwargs[field.name] = _CONFIG_CASTS[str(field.type)](data[key])

        state = {
            key.removeprefix("net."): data[key] for key in data.files if key.startswith("net.")
        }

    if not state:
        raise ValueError(
            f"{path} has no 'net.<param>' entries; a sequence-model file carries one "
            f"per network parameter"
        )
    return SequenceModel.load(state, SequenceConfig(**kwargs))
