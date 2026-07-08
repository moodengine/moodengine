"""Internal validation helpers shared by the public entry points.

Numeric backends (UMAP, sklearn, BLAS) either crash deep in their own stack
("Input contains NaN") without naming the offending data, or silently propagate
non-finite values into every downstream score. Checking at the moodengine
boundary costs a single O(n·d) pass — negligible next to any model or
clustering step — and produces an error message that names the array and the
rows to inspect.
"""

from __future__ import annotations

import numpy as np


def ensure_finite_2d(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Return ``X`` as a float32 ``(n, d)`` array, rejecting NaN/Inf.

    Raises ``ValueError`` when ``X`` is not 2-D or contains non-finite values;
    the message reports how many entries are non-finite and the first offending
    row indices so the caller can trace them back to tracks. An empty ``(0, d)``
    array passes (degenerate inputs are each entry point's own contract).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features); got shape {X.shape}")

    finite = np.isfinite(X)
    if not finite.all():
        n_bad = int((~finite).sum())
        bad_rows = np.unique(np.nonzero(~finite)[0])[:5].tolist()
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s) (NaN/Inf); "
            f"first offending row(s): {bad_rows}"
        )
    return X
