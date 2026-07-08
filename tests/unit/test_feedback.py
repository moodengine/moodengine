"""Unit tests for moodengine.feedback — implicit weak labels. Pure-Python, torch-free, AAA."""

from __future__ import annotations

from assertpy import assert_that

from moodengine.feedback import (
    BASE_WEIGHTS,
    EARLY_SKIP_POS,
    aggregate_implicit,
    implicit_weight,
)


def test_positive_events_are_plus_one():
    assert_that(implicit_weight("complete", 1.0)).is_equal_to(1.0)
    assert_that(implicit_weight("replay", 0.0)).is_equal_to(1.0)


def test_exposure_events_are_neutral():
    assert_that(implicit_weight("play", 0.0)).is_equal_to(0.0)
    assert_that(implicit_weight("seek", 0.5)).is_equal_to(0.0)
    assert_that(implicit_weight("unknown-event", 0.5)).is_equal_to(0.0)  # unknown -> neutral


def test_early_skip_is_strong_negative_late_skip_near_neutral():
    early = implicit_weight("skip", 0.05)  # before EARLY_SKIP_POS
    boundary = implicit_weight("skip", EARLY_SKIP_POS)
    late = implicit_weight("skip", 0.9)
    end = implicit_weight("skip", 1.0)
    assert_that(early).is_equal_to(-1.0)  # full-strength negative
    assert_that(boundary).is_equal_to(-1.0)  # still full at the threshold
    assert_that(late).is_greater_than(-0.2)  # attenuated toward neutral
    assert_that(late).is_less_than(0.0)
    assert_that(end).is_equal_to(0.0)  # skip at the very end == neutral
    # monotone: the later the skip, the less negative
    assert_that(early).is_less_than_or_equal_to(late)


def test_skip_weight_never_below_minus_one():
    for pos in (0.0, 0.1, 0.19, 0.2, 0.5, 1.0):
        assert_that(implicit_weight("skip", pos)).is_greater_than_or_equal_to(-1.0)


def test_aggregate_bounds_within_unit_interval():
    # Saturation: even 100 completes cannot push the weight past 1.0.
    events = [("t1", "complete", 1.0)] * 100
    w = aggregate_implicit(events)["t1"]
    assert_that(w).is_greater_than(0.0)
    assert_that(w).is_less_than_or_equal_to(1.0)
    # ...and 100 early skips cannot push below -1.0.
    w_neg = aggregate_implicit([("t2", "skip", 0.0)] * 100)["t2"]
    assert_that(w_neg).is_greater_than_or_equal_to(-1.0)
    assert_that(w_neg).is_less_than(0.0)


def test_aggregate_monotone_in_completes_and_early_skips():
    one = aggregate_implicit([("t", "complete", 1.0)])["t"]
    three = aggregate_implicit([("t", "complete", 1.0)] * 3)["t"]
    assert_that(three).is_greater_than_or_equal_to(
        one
    )  # more completes -> weight up (non-decreasing)

    s1 = aggregate_implicit([("t", "skip", 0.0)])["t"]
    s3 = aggregate_implicit([("t", "skip", 0.0)] * 3)["t"]
    assert_that(s3).is_less_than_or_equal_to(s1)  # more early skips -> weight down


def test_aggregate_excludes_exposure_only_tracks():
    # A track seen only via play/seek has NO weighing event -> absent (never a fabricated 0).
    out = aggregate_implicit(
        [("exposed", "play", 0.0), ("exposed", "seek", 0.4), ("liked", "complete", 1.0)]
    )
    assert_that(out).contains_key("liked")
    assert_that(out).does_not_contain_key("exposed")


def test_aggregate_empty_is_empty():
    assert_that(aggregate_implicit([])).is_equal_to({})


def test_aggregate_is_deterministic_and_order_independent():
    a = [("t", "complete", 1.0), ("t", "skip", 0.05), ("u", "replay", 0.0)]
    b = list(reversed(a))
    assert_that(aggregate_implicit(a)).is_equal_to(aggregate_implicit(b))


def test_base_weights_shape():
    assert_that(set(BASE_WEIGHTS)).is_equal_to({"complete", "replay", "play", "seek", "skip"})
