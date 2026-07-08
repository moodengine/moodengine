"""Shared numeric primitives used across the engine modules.

Single home for helpers that would otherwise be re-implemented per module.
``l2_normalize`` is also re-exported by :mod:`moodengine.pooling` and
:mod:`moodengine.labeling`, which are its historical public locations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> NDArray[np.float32]:
    """L2-normalize along ``axis``; safe for zero vectors (``eps`` floor).

    Input is cast to float32 and the result is float32. The norm is floored at
    ``eps`` before dividing, so a zero vector maps to a zero vector (never
    NaN/Inf) and a vector with norm below ``eps`` is scaled by ``1/eps`` rather
    than exactly normalized — the trade that keeps the operation finite.
    """
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    # copy=False: the division already produced a fresh float32 array.
    return (x / np.maximum(norm, eps)).astype(np.float32, copy=False)
