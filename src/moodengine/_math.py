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


#: Relative spread below which a series carries no correlation, only its own rounding error.
#: RELATIVE, not absolute: the affect axes live in [0, 1], but the same helper must not call a
#: genuinely small real spread constant. 1e-12 sits four orders above float64's relative epsilon
#: and eight below any spread these metrics are meant to score, so it separates noise from signal
#: without touching either.
_CONSTANT_REL_TOL: float = 1e-12


def is_constant_series(values: np.ndarray) -> bool:
    """``True`` when ``values`` has no spread beyond float rounding on its own scale.

    The guard every correlation here needs before dividing by a standard deviation. Testing
    ``std == 0.0`` only catches an EXACT tie: one ULP of difference passes it, and ``np.corrcoef``
    then returns a confident-looking correlation built entirely from representation error. The
    comparison is therefore against the magnitude the values sit on, not against zero.

    An empty or single-element input is constant by definition (nothing to vary).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 2:
        return True

    scale = float(np.max(np.abs(v)))
    return float(v.std()) <= _CONSTANT_REL_TOL * scale
