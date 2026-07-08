"""Tests for :mod:`moodengine.exceptions` — the failure hierarchy consumers catch.

The contract under test: one root (``MoodengineError``) covers every
library-specific failure; each subclass keeps the stdlib type it replaced as a
secondary base so pre-hierarchy catchers keep working; the missing-dependency
message always spells out the exact install command.
"""

from __future__ import annotations

import pytest
from assertpy import assert_that

from moodengine.exceptions import (
    AudioDecodeError,
    MissingDependencyError,
    ModelLoadError,
    MoodengineError,
)


def test_every_subclass_is_catchable_via_the_root() -> None:
    """One ``except MoodengineError`` covers all library failures."""
    for exc_type in (AudioDecodeError, MissingDependencyError, ModelLoadError):
        assert_that(issubclass(exc_type, MoodengineError)).is_true()


def test_subclasses_keep_their_pre_hierarchy_stdlib_bases() -> None:
    """Catchers written against the previous plain-stdlib raises keep working."""
    assert_that(issubclass(AudioDecodeError, RuntimeError)).is_true()  # was a bare RuntimeError
    assert_that(issubclass(MissingDependencyError, ImportError)).is_true()  # was a bare ImportError
    assert_that(issubclass(ModelLoadError, RuntimeError)).is_true()


def test_missing_dependency_message_spells_out_the_install_command() -> None:
    """The message names the feature, the package and the exact pip extra."""
    err = MissingDependencyError("ot_morph", "POT", "ot")

    assert_that(str(err)).is_equal_to('ot_morph requires POT: pip install "moodengine[ot]"')
    assert_that(err.feature).is_equal_to("ot_morph")
    assert_that(err.package).is_equal_to("POT")
    assert_that(err.extra).is_equal_to("ot")


def test_missing_dependency_optional_hint_is_appended() -> None:
    """A hint (e.g. a dependency-free alternative) lands in the message."""
    err = MissingDependencyError("backend='treeshap'", "shap", "explain", hint="use 'exact'")

    assert_that(str(err)).ends_with("(use 'exact')")
    assert_that(str(err)).contains('pip install "moodengine[explain]"')


def test_hierarchy_is_importable_from_the_package_root() -> None:
    """Consumers catch these from ``moodengine`` directly, not a private module."""
    import moodengine

    assert_that(moodengine.MoodengineError).is_same_as(MoodengineError)
    assert_that(moodengine.AudioDecodeError).is_same_as(AudioDecodeError)
    assert_that(moodengine.MissingDependencyError).is_same_as(MissingDependencyError)
    assert_that(moodengine.ModelLoadError).is_same_as(ModelLoadError)


def test_projection_unavailable_joins_the_hierarchy() -> None:
    """The pre-existing projection error is now a MissingDependencyError too —
    while keeping its original RuntimeError base and ``.method`` attribute."""
    from moodengine.cluster import ProjectionMethodUnavailable

    err = ProjectionMethodUnavailable("pacmap")

    assert_that(err).is_instance_of(MoodengineError)
    assert_that(err).is_instance_of(MissingDependencyError)
    assert_that(err).is_instance_of(RuntimeError)
    assert_that(err.method).is_equal_to("pacmap")
    assert_that(str(err)).contains('pip install "moodengine[pacmap]"')


def test_catching_the_root_catches_a_raised_subclass() -> None:
    with pytest.raises(MoodengineError, match=r"Failed to decode audio file: x\.mp3"):
        raise AudioDecodeError("Failed to decode audio file: x.mp3")
