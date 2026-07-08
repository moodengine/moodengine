"""Exception hierarchy: every failure a caller can act on has a catchable type.

``MoodengineError`` is the single root, so a consumer can wrap a pipeline call
in ``except MoodengineError`` and react per subclass instead of resorting to a
blind ``except Exception``. The subclasses map to failure families with
genuinely different remediations: an audio file that cannot be decoded (skip it
and move on), an optional backend that is not installed (install an extra and
retry), a model checkpoint that cannot be fetched or loaded (fix the
network/cache environment and retry).

Plain argument errors (unknown mode string, wrong shape, invalid value) stay
``ValueError``/``TypeError`` on purpose — that is the numpy/sklearn convention
generic numeric code already handles; wrapping those would only break it.
"""

from __future__ import annotations


class MoodengineError(Exception):
    """Root of every moodengine-specific failure."""


class AudioDecodeError(MoodengineError, RuntimeError):
    """An existing audio file could not be decoded (corrupt data, unsupported codec).

    Deliberately distinct from ``FileNotFoundError`` (raised separately for a
    missing path) so a caller can tell "re-scan the library" apart from "this
    file is damaged". Keeps ``RuntimeError`` as a secondary base because decode
    failures were raised as plain ``RuntimeError`` before this hierarchy existed.
    """


class MissingDependencyError(MoodengineError, ImportError):
    """A feature needs an optional backend that is not installed.

    ``feature`` names what was attempted, ``package`` the distribution that is
    missing and ``extra`` the pip extra that provides it; the message spells out
    the exact install command. Inherits ``ImportError`` so callers that caught
    the previous bare ``ImportError`` keep working.
    """

    def __init__(self, feature: str, package: str, extra: str, hint: str = "") -> None:
        message = f'{feature} requires {package}: pip install "moodengine[{extra}]"'
        if hint:
            message = f"{message} ({hint})"
        super().__init__(message)
        self.feature = feature
        self.package = package
        self.extra = extra


class ModelLoadError(MoodengineError, RuntimeError):
    """A model checkpoint could not be downloaded or loaded.

    The underlying huggingface/torch errors surface deep stack traces that never
    say which artifact was being fetched, so raise sites build a message that
    names the exact model/repo and how to pre-download it (e.g. with
    ``huggingface-cli download <repo>``; ``HF_HOME`` chooses the cache location
    and ``HF_HUB_OFFLINE=1`` forces cache-only resolution).
    """
