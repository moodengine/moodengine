"""Tip-Adapter (ECCV 2022) — a training-free key-value cache that turns confirmed few-shot examples
into an additive stream of learned logits, in the ProtoNet spirit.

Pure numpy, torch-free, deterministic. Given query embeddings ``X``, a cache of key embeddings ``K``
(the confirmed tracks) and one-hot values ``V`` (their corrected labels), it returns per-query
affinities ``A = exp(-beta * (1 - X @ K.T)) @ V`` — high where a query is cosine-close to keys of a
given label. Callers blend ``A`` additively BEFORE the softmax. An empty cache yields zeros, so
cold-start has exactly no effect. Every entry point guards empty matrices rather than raising.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from moodengine._math import l2_normalize as _l2_normalize
from moodengine._validation import ensure_finite_2d

DEFAULT_BETA: float = 5.0
DEFAULT_ALPHA: float = 1.0

_EPS: float = 1e-8
_LOGIT_CLAMP: float = 6.0  # finite saturated logit for a degenerate (single-class) OvR column


def tip_adapter_affinities(
    X: np.ndarray, K: np.ndarray, V: np.ndarray, beta: float = DEFAULT_BETA
) -> np.ndarray:
    """Tip-Adapter learned affinities ``A = exp(-beta * (1 - X @ K.T)) @ V``.

    ``X`` ``(n, d)`` query embeddings; ``K`` ``(m, d)`` cached key embeddings (defensively
    re-L2-normalized like the queries, so the dot products are cosines in ``[-1, 1]``); ``V``
    ``(m, L)`` one-hot label values. Returns ``A`` ``(n, L)`` float32, ``A >= 0`` by construction. An
    empty cache (``m == 0``), empty ``X``, or zero-width ``V`` yields ``np.zeros((n, L))`` — cold-start
    has exactly no effect. Inputs are never mutated; fully deterministic.
    """
    X = np.asarray(X, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)
    n = X.shape[0] if X.ndim == 2 else 0
    L = V.shape[1] if V.ndim == 2 else 0
    if n == 0 or K.ndim != 2 or K.shape[0] == 0 or L == 0:
        return np.zeros((n, L), dtype=np.float32)
    Xn = _l2_normalize(X, axis=1)
    Kn = _l2_normalize(K, axis=1)
    affinity = np.exp(-beta * (1.0 - Xn @ Kn.T))  # (n, m), each in (0, 1]
    return np.asarray(affinity @ V, dtype=np.float32)  # (n, L), >= 0


def prototype_vector(embs: np.ndarray) -> np.ndarray:
    """Few-shot mood **prototype**: the L2-normalized centroid of chosen tracks' CLAP embeddings.

    ``embs`` ``(m, d)`` are the CLAP embeddings of the ``m`` tracks a user picked to *define* a
    personal mood. Returns ``l2_normalize(embs.mean(axis=0))`` ``(d,)`` float32 — one query vector
    in the shared CLAP space, rankable by the same exact cosine kNN as any mood
    (``search.find_similar`` / an arbitrary-vector top-k). This is the ProtoNet prototype and the
    ``m``-example case of the Tip-Adapter key cache (:func:`tip_adapter_affinities`): a confirmed
    cluster collapsed to its mean direction. The final normalize is defensive — it rescales a mean
    whose norm shrank because the members point in spread-out directions. An empty selection
    (``m == 0``) yields ``np.zeros((d,))`` (``d`` inferred from ``embs`` when 2-D, else a length-0
    vector) — a null mood the caller drops. Inputs are never mutated; fully deterministic.
    """
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim != 2 or embs.shape[0] == 0:
        d = embs.shape[1] if embs.ndim == 2 else 0
        return np.zeros((d,), dtype=np.float32)
    return _l2_normalize(embs.mean(axis=0), axis=-1).astype(np.float32)


def acquisition_scores(probs: np.ndarray, strategy: str = "entropy") -> np.ndarray:
    """Per-track acquisition (uncertainty) score for active learning — how much a human label would
    teach. ``probs`` ``(n, n_moods)`` are per-row softmax probabilities; returns ``(n,)`` float32.

    Two classic uncertainty-sampling criteria (Lewis & Gale 1994; Settles 2009):
      * ``"entropy"`` — Shannon entropy ``-Σ_j p_j·log p_j`` (natural log; ``p·log p → 0`` at ``p=0``),
        bounded ``[0, log n_moods]``, maximal at the uniform distribution (least certain).
      * ``"margin"`` — ``1 - (p_(1) - p_(2))`` on the two largest probabilities, bounded ``[0, 1]``,
        maximal when the top two moods are tied (least separable). Higher = more uncertain either way.

    Guards ``n == 0`` and ``n_moods ∈ {0, 1}`` (no uncertainty possible → zeros). Pure numpy,
    deterministic; inputs are not mutated."""
    P = np.asarray(probs, dtype=np.float32)
    n = P.shape[0] if P.ndim == 2 else 0
    n_moods = P.shape[1] if P.ndim == 2 else 0
    if n == 0 or n_moods <= 1:
        return np.zeros((n,), dtype=np.float32)
    if strategy == "margin":
        top2 = np.sort(P, axis=1)[:, -2:]  # ascending -> [:, -1]=p1, [:, -2]=p2
        return (1.0 - (top2[:, 1] - top2[:, 0])).astype(np.float32)
    logp = np.zeros_like(P)
    np.log(P, out=logp, where=P > 0)  # log p where p>0, else 0 -> p·log p = 0 there
    return (-(P * logp).sum(axis=1)).astype(np.float32)


def diverse_subset(X: np.ndarray, scores: np.ndarray, n: int, gamma: float = 1.0) -> list[int]:
    """Greedy facility-location / BADGE-style selection of ≤ ``n`` diverse, high-score rows.

    Starts at ``argmax(scores)``, then greedily adds the row maximizing
    ``scores_i + gamma·(1 - max_{j∈chosen} cos(x_i, x_j))`` — trading raw acquisition score against
    dissimilarity to what's already picked, so the batch does not pile up near one uncertain region.
    ``X`` ``(m, d)`` is assumed L2-normalized (so ``cos = X @ X.T``). ``gamma=0`` reduces exactly to the
    pure top-``n`` by score. Returns row indices into ``X`` (≤ ``n``). Guards ``n ≤ 0`` / empty ``X``.

    This APPROXIMATES BADGE (Ash et al., ICLR 2020, which clusters gradient embeddings of a trained
    head) with a coverage objective over the frozen CLAP embeddings — documented as an approximation
    because no trained head exists yet (that arrives with the linear probe). Deterministic."""
    X = np.asarray(X, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    m = X.shape[0] if X.ndim == 2 else 0
    k = min(int(n), m)
    if k <= 0 or m == 0:
        return []
    chosen = [int(np.argmax(scores))]
    max_sim = X @ X[chosen[0]]  # (m,) cosine to the chosen set so far
    while len(chosen) < k:
        objective = scores + gamma * (1.0 - max_sim)
        objective[chosen] = -np.inf  # never re-pick
        nxt = int(np.argmax(objective))
        chosen.append(nxt)
        max_sim = np.maximum(max_sim, X @ X[nxt])
    return chosen


@dataclass(frozen=True)
class ProbeHead:
    """A lightweight classifier learned on FROZEN CLAP features — the linear probe / MLP head.

    Holds the fitted parameters plus the ``mood_names`` the logit columns align to and ``dim`` (the
    input width) as an alignment guard. ``method='linear'`` uses ``W`` ``(n_moods, d)`` / ``b``
    ``(n_moods,)`` (``hidden`` is ``None``); ``method='mlp'`` carries a one-hidden-layer forward pass
    in ``hidden = (W1 (d,h), b1 (h,), W2 (h,n_moods), b2 (n_moods,))`` and leaves ``W``/``b`` zeroed
    (unused). Pure data — no I/O, no algorithm; :func:`predict_probe` reads it,
    :func:`probe_state` / :func:`save_probe` persist it."""

    mood_names: list[str]
    W: np.ndarray  # (n_moods, d) OvR linear weights; float32
    b: np.ndarray  # (n_moods,) biases; float32
    method: str  # "linear" | "mlp"
    hidden: tuple[np.ndarray, ...] | None  # (W1, b1, W2, b2) for "mlp", else None
    dim: int  # d — input-width guard for predict_probe


def fit_linear_probe(
    X: np.ndarray,
    Y: np.ndarray,
    mood_names: list[str],
    *,
    method: str = "linear",
    C: float = 1.0,
    seed: int = 0,
) -> ProbeHead:
    """Fit a One-vs-Rest multi-label classifier on frozen CLAP embeddings — the linear probe.

    This is the canonical transfer baseline on frozen features: a *linear probe on frozen CLIP
    features* (Radford et al., ICML 2021) generalized to multi-label with an independent logit per
    mood (OvR). ``method='mlp'`` swaps the linear head for a one-hidden-layer MLP — the *CLIP-Adapter*
    (Gao et al., 2021) residual-on-frozen-features idea, here as a small trainable head whose logits
    the caller blends onto the recentered zero-shot prior. Unlike the training-free Tip-Adapter cache
    (:func:`tip_adapter_affinities`, which interpolates confirmed examples), the probe *generalizes*
    beyond the gold set.

    ``X`` ``(n, d)`` are L2-normalized frozen CLAP embeddings; ``Y`` ``(n, n_moods)`` are multi-hot
    OvR targets in ``{0,1}`` aligned to ``mood_names``. ``C`` is the inverse L2-regularization
    strength; ``seed`` fixes the (torch) MLP init. Returns a :class:`ProbeHead` whose logits align to
    ``mood_names``.

    ``linear`` is fit with ``sklearn.linear_model.LogisticRegression`` (OvR, per mood; cross-platform
    wheels, deterministic — no ``random_state`` needed for the ``lbfgs`` solver) — no torch. A
    degenerate column (a mood present in every / no training row) gets a constant saturated bias
    (``±_LOGIT_CLAMP``) instead of a fit. ``mlp`` imports ``torch`` LAZILY inside its branch only.
    Raises ``ValueError`` on ``n < 2``, ``n_moods < 2``, mis-shaped inputs, or a ``mood_names`` length
    mismatch. Inputs are never mutated."""
    Y = np.asarray(Y, dtype=np.float32)
    if np.asarray(X).ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D arrays")
    X = ensure_finite_2d(X, name="X")  # a NaN row would silently corrupt every mood's fit
    n, d = X.shape
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows")
    n_moods = Y.shape[1]
    if n < 2:
        raise ValueError("need at least 2 training examples")
    if n_moods < 2:
        raise ValueError("need at least 2 moods")
    if len(mood_names) != n_moods:
        raise ValueError("mood_names must align with Y columns")
    if method == "mlp":
        return _fit_probe_mlp(X, Y, list(mood_names), C=float(C), seed=int(seed))
    if method != "linear":
        raise ValueError(f"unknown method {method!r} (expected 'linear' | 'mlp')")
    return _fit_probe_linear(X, Y, list(mood_names), C=float(C))


def _fit_probe_linear(
    X: np.ndarray, Y: np.ndarray, mood_names: list[str], *, C: float
) -> ProbeHead:
    """OvR logistic regression per mood (numpy-in/out; torch-free)."""
    from sklearn.linear_model import LogisticRegression

    n, d = X.shape
    n_moods = len(mood_names)
    W = np.zeros((n_moods, d), dtype=np.float32)
    b = np.zeros((n_moods,), dtype=np.float32)
    for j in range(n_moods):
        yj = (Y[:, j] > 0.5).astype(np.int64)
        pos = int(yj.sum())
        if pos == 0:  # mood never positive -> constant "no" logit
            b[j] = np.float32(-_LOGIT_CLAMP)
            continue
        if pos == n:  # mood always positive -> constant "yes" logit
            b[j] = np.float32(_LOGIT_CLAMP)
            continue
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        clf.fit(X, yj)
        W[j] = clf.coef_[0].astype(np.float32)
        b[j] = np.float32(clf.intercept_[0])
    return ProbeHead(
        mood_names=list(mood_names), W=W, b=b, method="linear", hidden=None, dim=int(d)
    )


def _fit_probe_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    mood_names: list[str],
    *,
    C: float,
    seed: int,
    hidden_dim: int = 128,
    epochs: int = 300,
    lr: float = 1e-2,
) -> ProbeHead:
    """One-hidden-layer MLP head via torch (imported lazily — the ONLY torch use in this module).
    The fitted weights are returned as numpy so :func:`predict_probe` stays torch-free at inference."""
    import torch

    torch.manual_seed(int(seed))
    n, d = X.shape
    n_moods = len(mood_names)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    Yt = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))
    model = torch.nn.Sequential(
        torch.nn.Linear(d, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, n_moods),
    )
    weight_decay = 1.0 / (2.0 * max(C, _EPS) * n)  # L2 ∝ 1/C
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss_fn(model(Xt), Yt).backward()
        opt.step()
    model.eval()
    lin1, lin2 = model[0], model[2]
    hidden = (
        lin1.weight.detach().numpy().T.astype(np.float32),  # W1 (d, h)
        lin1.bias.detach().numpy().astype(np.float32),  # b1 (h,)
        lin2.weight.detach().numpy().T.astype(np.float32),  # W2 (h, n_moods)
        lin2.bias.detach().numpy().astype(np.float32),  # b2 (n_moods,)
    )
    return ProbeHead(
        mood_names=list(mood_names),
        W=np.zeros(
            (n_moods, d), dtype=np.float32
        ),  # unused for mlp; keeps the dim guard consistent
        b=np.zeros((n_moods,), dtype=np.float32),
        method="mlp",
        hidden=hidden,
        dim=int(d),
    )


def predict_probe(head: ProbeHead, X: np.ndarray) -> np.ndarray:
    """OvR logits ``(n, n_moods)`` for ``X`` under a fitted :class:`ProbeHead`, aligned to
    ``head.mood_names``. Pure numpy for BOTH methods (the MLP forward pass reuses the stored numpy
    weights, so inference never imports torch). Guards ``X.shape[1] == head.dim`` (``ValueError``
    otherwise). Higher logit = stronger evidence for that mood; the caller blends these onto the
    recentered zero-shot prior BEFORE the softmax. Inputs are never mutated."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    if X.shape[1] != head.dim:
        raise ValueError(f"X dim {X.shape[1]} != head dim {head.dim}")
    if head.method == "mlp":
        if head.hidden is None:
            raise ValueError("mlp head is missing its hidden weights")
        W1, b1, W2, b2 = head.hidden
        h = np.maximum(X @ W1 + b1, 0.0)  # ReLU
        return np.asarray(h @ W2 + b2, dtype=np.float32)
    return np.asarray(X @ head.W.T + head.b, dtype=np.float32)


# --- metric adapter: a learned linear projection of the CLAP space ----------------------------
# The Tip-Adapter cache and the linear probe improve CLASSIFICATION on the FROZEN CLAP geometry
# without ever moving it — similar / neighbours / search stay raw cosine on X_clap. The metric
# adapter is the capstone: it LEARNS the geometry itself. A tiny linear map g(x) = normalize(W·x),
# trained in Supervised Contrastive (SupCon, Khosla et al., NeurIPS 2020) on the mood gold, pulls
# same-mood tracks together and pushes different moods apart. The projected space `clap_adapted`
# improves BOTH the triptych (classification) AND cosine retrieval (similar/neighbours) — proven by
# a double metric downstream. Only fit_* touches torch; apply_projection is pure numpy.


@dataclass(frozen=True)
class Projection:
    """A learned linear projection of the CLAP space — the metric adapter's fitted parameters.

    ``g(x) = l2_normalize(W · x)`` maps a ``dim_in`` CLAP embedding to a ``dim_out`` point on the
    unit sphere where same-mood tracks are closer. Pure data — no I/O, no torch: :func:`apply_projection`
    reads it, :func:`projection_state` / :func:`save_projection` persist it. ``dim_in`` guards
    row-width alignment on apply; ``mood_names``
    records the contrastive classes at fit time (traceability); ``method`` is the training objective."""

    W: np.ndarray  # (dim_out, dim_in) float32 — linear projection, no bias
    dim_in: int  # = X.shape[1] at fit; apply-time alignment guard
    dim_out: int  # projected dimensionality (default = dim_in — a square projection)
    method: str  # "supcon" | "triplet"
    mood_names: list[str]  # contrastive classes (moods) at fit — provenance only


def apply_projection(proj: Projection, X: np.ndarray) -> np.ndarray:
    """Project + L2-normalize: ``l2_normalize(X @ proj.W.T)`` → ``(n, dim_out)`` float32. Pure numpy,
    torch-free, deterministic; inputs are never mutated.

    The final L2-normalization is the SAME one the triptych relies on (mirrors
    :func:`moodengine.labeling.l2_normalize`), so the projected embeddings are unit vectors and a dot
    product in the projected space is again a cosine — the recentered-softmax and the exact cosine kNN
    both transfer unchanged. Guards ``X.shape[1] == proj.dim_in`` (``ValueError`` otherwise); an empty
    ``X`` yields ``(0, dim_out)``."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    if X.shape[1] != proj.dim_in:
        raise ValueError(f"X dim {X.shape[1]} != projection dim_in {proj.dim_in}")
    if X.shape[0] == 0:
        return np.zeros((0, proj.dim_out), dtype=np.float32)
    return _l2_normalize(X @ proj.W.T, axis=1).astype(np.float32)


def fit_supcon_projection(
    X: np.ndarray,
    labels: np.ndarray,
    mood_names: list[str],
    *,
    dim_out: int | None = None,
    method: str = "supcon",
    temperature: float = 0.07,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 0,
) -> Projection:
    """Learn a linear metric adapter ``g(x) = normalize(W·x)`` over FROZEN CLAP embeddings.

    Trains a bias-free ``nn.Linear(dim_in, dim_out)`` so that, on the unit sphere of projected
    embeddings ``z = g(x)``, tracks sharing a mood are close and different moods are far:

      * ``method='supcon'`` — Supervised Contrastive loss (Khosla et al., NeurIPS 2020), the
        out-form ``L^sup_out``: for each anchor ``i`` with positives ``P(i)`` (same mood, self
        excluded), ``-1/|P(i)| Σ_{p∈P(i)} log( exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) )``. This is
        the label-supervised generalization of NT-Xent; the out-form is the paper's recommended,
        more stable variant. ``τ`` is the temperature.
      * ``method='triplet'`` — a batch-hard triplet margin loss in the FaceNet spirit (Schroff et
        al., CVPR 2015): per anchor, the hardest positive (least similar same-mood) must beat the
        hardest negative (most similar other-mood) by a cosine margin. Deterministic, no online
        semi-hard mining state.

    ``X`` ``(n, d)`` are L2-normalized frozen CLAP embeddings; ``labels`` ``(n,)`` are integer mood
    indices (the contrastive classes — the mood gold). ``dim_out`` defaults to ``d`` (a square
    projection preserving dimensionality). Full-batch gradient descent (the gold is small); Adam at
    ``lr`` for ``epochs`` steps. Deterministic at fixed ``seed`` (``torch.manual_seed`` seeds the
    Linear init) — two fits with the same seed give ``np.allclose`` weights. ``torch`` is imported
    LAZILY here (the only torch use in this module); the returned :class:`Projection` is pure numpy so
    :func:`apply_projection` stays torch-free.

    Raises ``ValueError`` on ``n < 2``, fewer than 2 distinct classes, mis-shaped inputs, or a class
    with no same-class partner (SupCon needs ≥2 examples per anchored class). Inputs are not mutated."""
    import torch

    y = np.asarray(labels).reshape(-1)
    X = ensure_finite_2d(X, name="X")  # a NaN row would silently corrupt the learned metric
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError("labels must align with X rows")
    if n < 2:
        raise ValueError("need at least 2 training examples")
    classes = np.unique(y)
    if classes.shape[0] < 2:
        raise ValueError("need at least 2 distinct classes")
    if method not in ("supcon", "triplet"):
        raise ValueError(f"unknown method {method!r} (expected 'supcon' | 'triplet')")
    d_out = int(dim_out) if dim_out else d

    torch.manual_seed(int(seed))
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64)))
    proj = torch.nn.Linear(d, d_out, bias=False)
    opt = torch.optim.Adam(proj.parameters(), lr=float(lr))

    # Same-class mask (n, n), self excluded — the positive set P(i) for every anchor.
    eye = torch.eye(n)
    same = (yt[:, None] == yt[None, :]).float()
    pos_mask = same * (1.0 - eye)
    not_self = 1.0 - eye
    tau = max(float(temperature), 1e-6)

    proj.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        z = torch.nn.functional.normalize(proj(Xt), dim=1)  # (n, d_out) on the unit sphere
        if method == "supcon":
            loss = _supcon_loss(z, pos_mask, not_self, tau)
        else:
            loss = _triplet_loss(z, same, eye)
        loss.backward()
        opt.step()

    W = proj.weight.detach().numpy().astype(np.float32)  # (d_out, d)
    return Projection(
        W=W, dim_in=int(d), dim_out=int(d_out), method=method, mood_names=list(mood_names)
    )


def _supcon_loss(z, pos_mask, not_self, tau: float):
    """SupCon L^sup_out over projected embeddings ``z`` (torch). ``pos_mask``/``not_self`` are the
    same-class (self-excluded) and off-diagonal masks. Anchors with no positive in the batch are
    dropped from the mean (they carry no supervised signal), not forced to a fabricated 0."""
    import torch

    sims = (z @ z.t()) / tau  # (n, n)
    sims = sims - sims.max(dim=1, keepdim=True).values.detach()  # row-wise stabilization
    exp_sims = torch.exp(sims) * not_self  # exclude self from the denominator
    log_prob = sims - torch.log(exp_sims.sum(dim=1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(dim=1)  # |P(i)|
    valid = pos_count > 0
    if not bool(valid.any()):
        raise ValueError("SupCon needs at least one class with ≥2 examples")
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()


def _triplet_loss(z, same, eye, margin: float = 0.2):
    """Batch-hard cosine triplet loss (FaceNet spirit) over projected ``z`` (torch). Per anchor: the
    hardest positive (min same-class cosine) and hardest negative (max other-class cosine); hinge at
    ``margin``. Anchors lacking a positive OR a negative in the batch are skipped."""
    import torch

    cos = z @ z.t()  # (n, n) cosine (z is unit-norm)
    pos = same * (1.0 - eye)
    neg = 1.0 - same  # different class (diagonal is same → 0)
    has_pos = pos.sum(dim=1) > 0
    has_neg = neg.sum(dim=1) > 0
    valid = has_pos & has_neg
    if not bool(valid.any()):
        raise ValueError("triplet loss needs anchors with both a positive and a negative")
    # Hardest positive = smallest same-class cosine (mask others to +inf); hardest negative = largest
    # other-class cosine (mask others to -inf).
    hardest_pos = torch.where(pos > 0, cos, torch.full_like(cos, float("inf"))).min(dim=1).values
    hardest_neg = torch.where(neg > 0, cos, torch.full_like(cos, float("-inf"))).max(dim=1).values
    losses = torch.clamp(hardest_neg - hardest_pos + margin, min=0.0)
    return losses[valid].mean()


# --- persistence: canonical state mapping + opt-in npz save/load ------------------------------
# ProbeHead and Projection are pure data that the CALLER persists. Without one canonical
# array-only mapping, every consumer hand-rolls the field-by-field (de)serialization and the
# copies drift silently the day a field is added. The *_state / *_from_state pair below is that
# single seam — plain ``np.ndarray`` values under stable string keys, storable in any array
# store or language. save_* / load_* are thin npz conveniences for file-based callers; they are
# separate functions because compute functions never write to disk (I/O stays opt-in).

_PROBE_SCHEMA: str = "moodengine.probe/1"
_PROJECTION_SCHEMA: str = "moodengine.projection/1"

_PROBE_REQUIRED_KEYS: tuple[str, ...] = (
    "schema",
    "mood_names",
    "W",
    "b",
    "method",
    "dim",
    "has_hidden",
)
_PROBE_HIDDEN_KEYS: tuple[str, ...] = ("hidden0", "hidden1", "hidden2", "hidden3")
_PROJECTION_REQUIRED_KEYS: tuple[str, ...] = (
    "schema",
    "W",
    "dim_in",
    "dim_out",
    "method",
    "mood_names",
)


def probe_state(head: ProbeHead) -> dict[str, np.ndarray]:
    """Canonical array-only state of a :class:`ProbeHead` — the serialization seam.

    Every value is a plain ``np.ndarray`` under a stable string key, so the mapping round-trips
    through any array store (npz, zarr, a database column, another language). Keys:

    * ``"schema"`` — 0-d unicode, always ``"moodengine.probe/1"`` (format version tag).
    * ``"mood_names"`` — ``(n_moods,)`` unicode, the names the logit columns align to.
    * ``"W"`` — ``(n_moods, dim)`` float32, C-contiguous One-vs-Rest weights.
    * ``"b"`` — ``(n_moods,)`` float32 biases.
    * ``"method"`` — 0-d unicode, ``"linear"`` or ``"mlp"``.
    * ``"dim"`` — 0-d int, the input width :func:`predict_probe` guards against.
    * ``"has_hidden"`` — 0-d bool, whether the four MLP arrays follow.
    * ``"hidden0"`` .. ``"hidden3"`` — float32 ``W1 (dim, h)``, ``b1 (h,)``, ``W2 (h, n_moods)``,
      ``b2 (n_moods,)``, in that order; present only when ``head.hidden`` is not ``None``.

    The head is never mutated. Round-trip law: ``probe_from_state(probe_state(h))`` equals ``h``
    field for field, arrays byte-identical."""
    state: dict[str, np.ndarray] = {
        "schema": np.array(_PROBE_SCHEMA),
        "mood_names": np.asarray(head.mood_names, dtype=np.str_),
        "W": np.ascontiguousarray(head.W, dtype=np.float32),
        "b": np.ascontiguousarray(head.b, dtype=np.float32),
        "method": np.array(head.method),
        "dim": np.array(head.dim, dtype=np.int64),
        "has_hidden": np.array(head.hidden is not None),
    }

    if head.hidden is not None:
        for i, layer in enumerate(head.hidden):
            state[f"hidden{i}"] = np.ascontiguousarray(layer, dtype=np.float32)

    return state


def probe_from_state(state: Mapping[str, np.ndarray]) -> ProbeHead:
    """Rebuild a :class:`ProbeHead` from a :func:`probe_state` mapping (key inventory there).

    Validates before trusting: missing required keys, a schema tag other than
    ``"moodengine.probe/1"``, a ``method`` outside ``'linear' | 'mlp'``, ``W``/``b`` shapes misaligned
    with ``mood_names``, a ``linear`` ``W`` whose width differs from ``dim``, or an ``mlp`` state
    lacking ``hidden0..hidden3`` or carrying mis-shaped ones — all raise ``ValueError`` stating what was received and
    what was expected — a stale or hand-built state fails loudly here, not as a silent
    misprediction downstream. Arrays are decoded as float32; the mapping is never mutated."""
    missing = [key for key in _PROBE_REQUIRED_KEYS if key not in state]
    if missing:
        raise ValueError(
            f"probe state is missing keys {missing} (expected the {_PROBE_SCHEMA!r} layout)"
        )

    schema = str(np.asarray(state["schema"]))
    if schema != _PROBE_SCHEMA:
        raise ValueError(f"unknown probe state schema {schema!r} (expected {_PROBE_SCHEMA!r})")

    mood_names = [str(name) for name in np.asarray(state["mood_names"]).reshape(-1)]
    W = np.asarray(state["W"], dtype=np.float32)
    b = np.asarray(state["b"], dtype=np.float32)
    method = str(np.asarray(state["method"]))
    dim = int(np.asarray(state["dim"]))
    has_hidden = bool(np.asarray(state["has_hidden"]))

    if W.ndim != 2:
        raise ValueError(f"W must be 2-D, got shape {W.shape}")
    if W.shape[0] != len(mood_names):
        raise ValueError(
            f"W has {W.shape[0]} rows but mood_names lists {len(mood_names)} moods "
            "(expected one weight row per mood)"
        )
    if b.shape != (len(mood_names),):
        raise ValueError(
            f"b has shape {b.shape} but mood_names lists {len(mood_names)} moods "
            "(expected one bias per mood)"
        )
    if method not in ("linear", "mlp"):
        raise ValueError(f"unknown method {method!r} (expected 'linear' | 'mlp')")

    hidden: tuple[np.ndarray, ...] | None = None
    if method == "mlp":
        if not has_hidden:
            raise ValueError(
                "method is 'mlp' but has_hidden is False (expected hidden0..hidden3 present)"
            )
        missing_hidden = [key for key in _PROBE_HIDDEN_KEYS if key not in state]
        if missing_hidden:
            raise ValueError(
                f"mlp probe state is missing keys {missing_hidden} "
                "(expected hidden0..hidden3 = W1, b1, W2, b2)"
            )
        hidden = tuple(np.asarray(state[key], dtype=np.float32) for key in _PROBE_HIDDEN_KEYS)
        if hidden[0].ndim != 2:
            raise ValueError(f"hidden0 (W1) must be 2-D, got shape {hidden[0].shape}")
        if hidden[0].shape[0] != dim:
            raise ValueError(
                f"hidden0 (W1) has {hidden[0].shape[0]} input rows but dim is {dim} "
                "(expected W1 of shape (dim, hidden_width))"
            )
        h_width = hidden[0].shape[1]
        n_moods = len(mood_names)
        if hidden[1].shape != (h_width,):
            raise ValueError(
                f"hidden1 (b1) has shape {hidden[1].shape} "
                f"(expected ({h_width},), one bias per hidden unit of W1)"
            )
        if hidden[2].shape != (h_width, n_moods):
            raise ValueError(
                f"hidden2 (W2) has shape {hidden[2].shape} (expected ({h_width}, {n_moods}))"
            )
        if hidden[3].shape != (n_moods,):
            raise ValueError(
                f"hidden3 (b2) has shape {hidden[3].shape} "
                f"(expected ({n_moods},), one bias per mood)"
            )
    elif W.shape[1] != dim:
        raise ValueError(
            f"W has {W.shape[1]} columns but dim is {dim} (expected one column per input feature)"
        )

    return ProbeHead(mood_names=mood_names, W=W, b=b, method=method, hidden=hidden, dim=dim)


def projection_state(proj: Projection) -> dict[str, np.ndarray]:
    """Canonical array-only state of a :class:`Projection` — the serialization seam.

    Every value is a plain ``np.ndarray`` under a stable string key, so the mapping round-trips
    through any array store (npz, zarr, a database column, another language). Keys:

    * ``"schema"`` — 0-d unicode, always ``"moodengine.projection/1"`` (format version tag).
    * ``"W"`` — ``(dim_out, dim_in)`` float32, C-contiguous linear map (no bias).
    * ``"dim_in"`` / ``"dim_out"`` — 0-d ints, the widths :func:`apply_projection` relies on.
    * ``"method"`` — 0-d unicode, the training objective (provenance, e.g. ``"supcon"``).
    * ``"mood_names"`` — ``(n_classes,)`` unicode, the contrastive classes at fit time; may be
      empty (provenance only, never read at apply time).

    The projection is never mutated. Round-trip law: ``projection_from_state(projection_state(p))``
    equals ``p`` field for field, ``W`` byte-identical."""
    return {
        "schema": np.array(_PROJECTION_SCHEMA),
        "W": np.ascontiguousarray(proj.W, dtype=np.float32),
        "dim_in": np.array(proj.dim_in, dtype=np.int64),
        "dim_out": np.array(proj.dim_out, dtype=np.int64),
        "method": np.array(proj.method),
        "mood_names": np.asarray(proj.mood_names, dtype=np.str_),
    }


def projection_from_state(state: Mapping[str, np.ndarray]) -> Projection:
    """Rebuild a :class:`Projection` from a :func:`projection_state` mapping (key inventory there).

    Validates before trusting: missing required keys, a schema tag other than
    ``"moodengine.projection/1"``, a ``W`` whose shape is not exactly ``(dim_out, dim_in)``, or an
    empty ``method`` all raise ``ValueError`` stating what was received and what was expected.
    ``method`` is provenance, so any NON-EMPTY string is accepted — a state written by a future
    objective still loads. ``W`` is decoded as float32; the mapping is never mutated."""
    missing = [key for key in _PROJECTION_REQUIRED_KEYS if key not in state]
    if missing:
        raise ValueError(
            f"projection state is missing keys {missing} "
            f"(expected the {_PROJECTION_SCHEMA!r} layout)"
        )

    schema = str(np.asarray(state["schema"]))
    if schema != _PROJECTION_SCHEMA:
        raise ValueError(
            f"unknown projection state schema {schema!r} (expected {_PROJECTION_SCHEMA!r})"
        )

    W = np.asarray(state["W"], dtype=np.float32)
    dim_in = int(np.asarray(state["dim_in"]))
    dim_out = int(np.asarray(state["dim_out"]))
    method = str(np.asarray(state["method"]))
    mood_names = [str(name) for name in np.asarray(state["mood_names"]).reshape(-1)]

    if W.shape != (dim_out, dim_in):
        raise ValueError(
            f"W has shape {W.shape} but (dim_out, dim_in) is ({dim_out}, {dim_in}) "
            "(expected W of exactly that shape)"
        )
    if not method:
        raise ValueError(
            "method is empty (expected a non-empty provenance string such as 'supcon')"
        )

    return Projection(W=W, dim_in=dim_in, dim_out=dim_out, method=method, mood_names=mood_names)


def save_probe(head: ProbeHead, path: str | Path) -> Path:
    """Write :func:`probe_state` as an uncompressed ``.npz`` archive at EXACTLY ``path``.

    ``np.savez`` silently appends a ``.npz`` suffix when handed a bare path, so the archive is
    written through an open file handle instead — the caller's filename is honored verbatim
    (returned as ``Path(path)`` for chaining). The parent directory must already exist: this
    function never creates directories (the caller owns its layout), so a missing parent raises
    the usual ``OSError`` from ``open``. This is the opt-in I/O companion to the pure state
    mapping; :func:`load_probe` reads it back."""
    out = Path(path)

    # No allow_pickle here: np.savez only grew that keyword in numpy 2.1, and on older
    # supported versions it would fall into **kwds and be WRITTEN into the archive as a bogus
    # 'allow_pickle' entry. probe_state emits plain arrays only, so nothing can pickle at
    # write time; the security guarantee (refusing pickle) lives on the load side. The
    # Any-typed dict keeps mypy from matching the ** expansion against that keyword.
    state: dict[str, Any] = dict(probe_state(head))
    with open(out, "wb") as f:
        np.savez(f, **state)

    return out


def load_probe(path: str | Path) -> ProbeHead:
    """Read a :func:`save_probe` archive back into a :class:`ProbeHead`.

    A missing file raises ``FileNotFoundError`` naming the path. The archive is opened with
    ``allow_pickle=False`` — the format is plain arrays, and refusing pickle means a tampered
    file cannot execute code on load. Content validation (schema tag, shapes, method) is
    delegated to :func:`probe_from_state`, which raises ``ValueError`` on any mismatch."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"probe state file not found: {path}")

    with np.load(p, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}

    return probe_from_state(state)


def save_projection(proj: Projection, path: str | Path) -> Path:
    """Write :func:`projection_state` as an uncompressed ``.npz`` archive at EXACTLY ``path``.

    ``np.savez`` silently appends a ``.npz`` suffix when handed a bare path, so the archive is
    written through an open file handle instead — the caller's filename is honored verbatim
    (returned as ``Path(path)`` for chaining). The parent directory must already exist: this
    function never creates directories (the caller owns its layout), so a missing parent raises
    the usual ``OSError`` from ``open``. This is the opt-in I/O companion to the pure state
    mapping; :func:`load_projection` reads it back."""
    out = Path(path)

    # No allow_pickle here — same rationale as save_probe: the keyword only exists from
    # numpy 2.1 and would otherwise pollute the archive on older supported versions.
    state: dict[str, Any] = dict(projection_state(proj))
    with open(out, "wb") as f:
        np.savez(f, **state)

    return out


def load_projection(path: str | Path) -> Projection:
    """Read a :func:`save_projection` archive back into a :class:`Projection`.

    A missing file raises ``FileNotFoundError`` naming the path. The archive is opened with
    ``allow_pickle=False`` — the format is plain arrays, and refusing pickle means a tampered
    file cannot execute code on load. Content validation (schema tag, ``W`` shape, method) is
    delegated to :func:`projection_from_state`, which raises ``ValueError`` on any mismatch."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"projection state file not found: {path}")

    with np.load(p, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}

    return projection_from_state(state)
