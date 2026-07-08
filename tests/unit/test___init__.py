"""Contract tests for the public API surface (``moodengine.__all__``).

The package re-exports its public names at the root; these tests pin the two
directions of that contract so imports and ``__all__`` cannot drift apart:
every advertised name resolves, and every name bound at the root is advertised.
"""

from __future__ import annotations

import types

from assertpy import assert_that

import moodengine


def test_every_all_entry_resolves_to_a_root_attribute() -> None:
    missing = [name for name in moodengine.__all__ if not hasattr(moodengine, name)]

    assert_that(missing).is_empty()


def test_all_has_no_duplicate_entries() -> None:
    assert_that(len(set(moodengine.__all__))).is_equal_to(len(moodengine.__all__))


def test_every_public_root_binding_is_advertised_in_all() -> None:
    # Submodules land in the namespace as a side effect of the re-export imports and are
    # not part of the advertised surface; ``annotations`` is the __future__ import's binding.
    public = {
        name
        for name, value in vars(moodengine).items()
        if not name.startswith("_")
        and not isinstance(value, types.ModuleType)
        and name != "annotations"
    }

    assert_that(sorted(public)).is_equal_to(sorted(set(moodengine.__all__) - {"__version__"}))
