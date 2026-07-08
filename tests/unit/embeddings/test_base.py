"""Unit tests for moodengine.embeddings.base — fingerprint + cache primitives.

Pure, torch-free. AAA + fluent assertions (assertpy).
"""

from __future__ import annotations

import numpy as np
from assertpy import assert_that

from moodengine.embeddings import base


def test_file_fingerprint_is_content_addressed(tmp_path):
    # Arrange
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"same-bytes")
    b.write_bytes(b"same-bytes")
    c = tmp_path / "c.bin"
    c.write_bytes(b"other-bytes")

    # Act
    fp_a, fp_b, fp_c = (base.file_fingerprint(p) for p in (a, b, c))

    # Assert
    assert_that(fp_a).is_equal_to(fp_b)  # same content -> same fingerprint
    assert_that(fp_a).is_not_equal_to(fp_c)  # different content -> different
    assert_that(fp_a).is_type_of(str).is_not_empty()


def test_cache_key_varies_with_model_and_extra(tmp_path):
    # Arrange
    f = tmp_path / "track.wav"
    f.write_bytes(b"audio")

    # Act
    k_mert = base.cache_key(f, "mert", extra="mean_std_seg10")
    k_clap = base.cache_key(f, "clap", extra="mean_std_seg10")
    k_mert_again = base.cache_key(f, "mert", extra="mean_std_seg10")

    # Assert
    assert_that(k_mert).is_equal_to(k_mert_again)  # deterministic
    assert_that(k_mert).is_not_equal_to(k_clap)  # model name is part of the key
    assert_that(k_mert).contains("mert")


def test_default_clap_embedding_key_is_byte_identical(monkeypatch, tmp_path):
    """Regression guard: the DEFAULT CLAP track-embedding cache key must
    stay exactly ``clap__<fp>__mean_std_seg10``. Any drift silently invalidates every on-disk
    ``.npy`` cache. The fingerprint is frozen so the assertion pins an exact literal."""
    # Arrange
    from moodengine.config import default_config

    cfg = default_config()
    extra = f"{cfg.pooling_mode}_seg{int(cfg.segment_seconds)}"
    monkeypatch.setattr(base, "file_fingerprint", lambda p: "deadbeefdeadbeef")
    f = tmp_path / "track.wav"
    f.write_bytes(b"audio")

    # Act
    key = base.cache_key(f, "clap", extra=extra)

    # Assert
    assert_that(extra).is_equal_to("mean_std_seg10")  # default config -> legacy tag
    assert_that(key).is_equal_to("clap__deadbeefdeadbeef__mean_std_seg10")


def test_provenance_cache_key_matches_triplet_format():
    """``provenance_cache_key`` spells the content-addressed triplet used by the provenance layer."""
    # Act / Assert
    assert_that(base.provenance_cache_key("clap", "0.1.0", "abcd", "ef01")).is_equal_to(
        "clap__0.1.0__abcd__ef01"
    )
    # Slashes in the model name are sanitized just like cache_key does.
    assert_that(base.provenance_cache_key("a/b", "0.1.0", "abcd", "ef01")).is_equal_to(
        "a_b__0.1.0__abcd__ef01"
    )


def test_save_then_load_roundtrips_array(tmp_path):
    # Arrange
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    key = "model__deadbeef"

    # Act
    base.save_cached(tmp_path, key, arr)
    loaded = base.load_cached(tmp_path, key)

    # Assert
    assert_that(loaded).is_not_none()
    np.testing.assert_array_equal(loaded, arr)


def test_load_cached_returns_none_on_miss(tmp_path):
    # Arrange / Act
    result = base.load_cached(tmp_path, "never-written")

    # Assert
    assert_that(result).is_none()


def test_load_cached_treats_corrupt_entry_as_miss_and_purges_it(tmp_path, caplog):
    """A truncated/corrupt .npy is a cache MISS, not an error: without this, one bad
    entry would silently make a perfectly decodable track vanish from every run."""
    # Arrange — 10 junk bytes are not a valid .npy header.
    key = "model__corrupt"
    corrupt = base.cache_path(tmp_path, key)
    corrupt.write_bytes(b"\x00garbage!!")

    # Act
    with caplog.at_level("WARNING", logger="moodengine.embeddings.base"):
        result = base.load_cached(tmp_path, key)

    # Assert — miss, logged, and the poison file is gone so the recompute can land.
    assert_that(result).is_none()
    assert_that(corrupt.exists()).is_false()
    assert_that(caplog.text).contains("Corrupt cache entry")


def test_corrupt_entry_recovers_on_the_next_save(tmp_path):
    """The full recovery path: corrupt entry → miss → recompute+save → hit."""
    # Arrange
    key = "model__recovers"
    base.cache_path(tmp_path, key).write_bytes(b"\x00garbage!!")
    arr = np.arange(6, dtype=np.float32)

    # Act — the miss clears the entry; a fresh save takes its place.
    assert_that(base.load_cached(tmp_path, key)).is_none()
    base.save_cached(tmp_path, key, arr)
    loaded = base.load_cached(tmp_path, key)

    # Assert
    np.testing.assert_array_equal(loaded, arr)


def test_save_cached_is_atomic_no_temp_residue(tmp_path):
    """The write goes through a temp file + os.replace: after saving, exactly the
    final .npy exists — no half-written target, no leftover temp file."""
    # Arrange
    arr = np.ones(4, dtype=np.float32)

    # Act
    base.save_cached(tmp_path, "model__atomic", arr)

    # Assert — only the final artifact remains.
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert_that(leftovers).is_equal_to(["model__atomic.npy"])
