"""Track-level pooling of frame/clip embeddings into a single L2-normalized vector.

Pure numpy, torch-free. The pipeline picks a pooler by embedder name via
:data:`POOLERS`. MERT produces frame-level hidden states per segment
(``(n_layers, n_frames, hidden)``) which are layer-combined, frame-pooled and
normalized; CLAP produces one clip embedding per segment (``(hidden,)``) which
are averaged and normalized.
"""

from __future__ import annotations

import numpy as np

from moodengine._math import l2_normalize
from moodengine._typing import LayerWeighting, PoolingMode
from moodengine.config import Config


def _stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    x = np.asarray(x, dtype=np.float32)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32, copy=False)


def weight_layers(
    frame_emb: np.ndarray,
    mode: LayerWeighting,
    layers: tuple[int, ...] | None = None,
    layer_weights: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Combine MERT layers into a single per-frame matrix.

    ``frame_emb``: ``(n_layers, n_frames, hidden)``. Returns ``(n_frames, hidden)``.

    ``mode``:
    - ``'uniform'``: mean over all layers.
    - ``'last'``: last layer only.
    - ``'subset'``: mean over ``layers`` (tuple of indices) if given, else the
      middle third of layers (``range(n//3, 2*n//3 + 1)``), clamped to valid range.
    - ``'weighted'``: ``softmax(layer_weights)``-weighted sum over layers;
      ``layer_weights`` length must equal ``n_layers`` — a missing or
      mis-sized tuple raises ``ValueError`` (a silent uniform fallback would
      return a numerically different result than the one asked for).
    """
    frame_emb = np.asarray(frame_emb, dtype=np.float32)
    if frame_emb.ndim != 3:
        raise ValueError(
            f"weight_layers expects (n_layers, n_frames, hidden), got shape {frame_emb.shape}"
        )
    n_layers = frame_emb.shape[0]
    if mode == "last":
        # astype WITHOUT copy=False on purpose: frame_emb[-1] is a view, and the
        # copy releases the full (n_layers, n_frames, hidden) stack it pins.
        return frame_emb[-1].astype(np.float32)
    if mode == "uniform":
        return frame_emb.mean(axis=0).astype(np.float32, copy=False)
    if mode == "subset":
        if layers:
            idx = sorted({int(i) for i in layers if 0 <= int(i) < n_layers})
            if not idx:
                idx = list(range(n_layers))
        else:
            idx = list(range(n_layers // 3, 2 * n_layers // 3 + 1))
            idx = [i for i in idx if 0 <= i < n_layers] or list(range(n_layers))
        return frame_emb[idx].mean(axis=0).astype(np.float32, copy=False)
    if mode == "weighted":
        if layer_weights is None or len(layer_weights) != n_layers:
            got = "None" if layer_weights is None else str(len(layer_weights))
            raise ValueError(
                f"layer_weights has {got} entries but the model produced {n_layers} layers; "
                f"mode='weighted' needs one logit per layer (it never falls back silently)"
            )
        w = _stable_softmax(np.asarray(layer_weights, dtype=np.float32))
        return np.tensordot(w, frame_emb, axes=([0], [0])).astype(np.float32, copy=False)
    raise ValueError(f"unknown layer weighting mode: {mode!r}")


def pool_frames(frames_2d: np.ndarray, mode: PoolingMode) -> np.ndarray:
    """Pool a ``(n_frames, hidden)`` matrix into a 1-D vector.

    ``mode`` ``'mean'`` -> ``(hidden,)``; ``'mean_std'`` -> ``concat[mean, std]``
    -> ``(2*hidden,)``.
    """
    frames_2d = np.asarray(frames_2d, dtype=np.float32)
    if frames_2d.ndim != 2:
        raise ValueError(f"pool_frames expects (n_frames, hidden), got shape {frames_2d.shape}")
    mean = frames_2d.mean(axis=0)
    if mode == "mean":
        return mean.astype(np.float32, copy=False)
    if mode == "mean_std":
        std = frames_2d.std(axis=0)
        return np.concatenate([mean, std]).astype(np.float32, copy=False)
    raise ValueError(f"unknown pooling mode: {mode!r}")


def pool_mert(segments: list[np.ndarray], config: Config) -> np.ndarray:
    """Pool MERT segment hidden states into one track vector.

    ``segments``: list of ``(n_layers, n_frames, hidden)`` (one per audio segment).
    Layer-combine each via :func:`weight_layers` (``config.mert_layer_weighting``),
    concatenate all frames across segments along the frame axis, pool via
    :func:`pool_frames` (``config.pooling_mode``), then L2-normalize. Returns a
    1-D float32 track vector.
    """
    if not segments:
        raise ValueError("pool_mert received no segments")
    per_segment_frames = [
        weight_layers(
            seg,
            config.mert_layer_weighting,
            layers=config.mert_layers,
            layer_weights=config.mert_layer_weights,
        )
        for seg in segments
    ]
    all_frames = np.concatenate(per_segment_frames, axis=0)
    pooled = pool_frames(all_frames, config.pooling_mode)
    return l2_normalize(pooled, axis=-1)


def pool_clap(segments: list[np.ndarray], config: Config) -> np.ndarray:
    """Pool CLAP clip embeddings into one track vector.

    ``segments``: list of ``(hidden,)`` clip embeddings (one per segment). Mean
    over segments, then L2-normalize. Returns a 1-D float32 track vector.
    (``config.pooling_mode`` is ignored for CLAP — clip embeddings have no frame
    axis.)
    """
    if not segments:
        raise ValueError("pool_clap received no segments")
    stacked = np.stack([np.asarray(s, dtype=np.float32).reshape(-1) for s in segments], axis=0)
    mean = stacked.mean(axis=0)
    return l2_normalize(mean, axis=-1)


# A registry so the pipeline can pick the pooler by embedder name:
POOLERS = {"mert": pool_mert, "clap": pool_clap}
