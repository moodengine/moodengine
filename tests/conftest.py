"""Shared, torch-free test doubles and audio synthesis for the moodengine suite.

Two families of shared tooling live here so no test file re-implements them:

* Fake embedders (``make_fake_embedder`` factory, ``fake_clap``) — drive the
  pipeline / labeling / search stages without constructing a real MERT/CLAP
  model, and therefore without torch.
* Audio synthesis (``synth_clip``, ``make_audio_library``) — short, deterministic,
  spectrally distinct clips written as real WAV files, so decode → segment →
  embed paths run against genuine audio instead of ad-hoc inline sines.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def hash_unit_vec(key: bytes, dim: int) -> np.ndarray:
    """Deterministic L2-normalized vector seeded by ``key`` (reproducible)."""
    seed = int.from_bytes(hashlib.sha1(key).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEmbedder:
    """Torch-free stand-in usable as a 'clap' or 'mert' embedder.

    ``extract`` derives a deterministic vector from the waveform content (distinct
    clips -> distinct points, so clustering is meaningful). For ``name='clap'`` it
    returns a ``(dim,)`` clip embedding (what ``pool_clap`` expects); for ``'mert'``
    a ``(n_layers, n_frames, hidden)`` tensor (what ``pool_mert`` expects).
    ``embed_text`` maps prompts to deterministic L2-normed rows.
    """

    def __init__(self, name: str, sample_rate: int, dim: int = 8) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self.dim = dim

    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        key = wav.tobytes()[:4096] + str(round(float(wav.sum()), 3)).encode()
        vec = hash_unit_vec(key, self.dim)
        if self.name == "mert":
            return np.stack([vec, vec], axis=0)[None, :, :].astype(np.float32)
        return vec

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        rows = [hash_unit_vec(("txt:" + p).encode(), self.dim) for p in prompts]
        return np.vstack(rows).astype(np.float32) if rows else np.empty((0, self.dim), np.float32)


# --------------------------------------------------------------------------- #
# Synthetic audio: short clips with genuinely different spectra
# --------------------------------------------------------------------------- #
CLIP_KINDS = ("tone", "chirp", "drone", "arpeggio", "percussive")


def _synth_clip(
    kind: str = "tone", seconds: float = 1.0, sr: int = 48_000, seed: int = 0
) -> np.ndarray:
    """Deterministic mono float32 clip of the requested character.

    Each kind occupies a distinct spectral region — sustained harmonic note
    (``tone``), rising sweep (``chirp``), low detuned beating pair (``drone``),
    bright major-triad steps (``arpeggio``), gated noise bursts (``percussive``)
    — so a set of clips clusters apart and embedders see real variety instead
    of n copies of the same sine. Same (kind, seed) → identical samples
    (seeded via a stable content hash, immune to per-process ``hash()`` salting).
    """
    digest = hashlib.sha1(f"{kind}:{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:4], "big"))
    t = np.linspace(0.0, seconds, int(round(sr * seconds)), endpoint=False)
    f0 = float(rng.uniform(130.0, 520.0))

    if kind == "tone":
        harmonics = enumerate((1.0, 0.5, 0.25, 0.12))
        wave = sum(a * np.sin(2 * np.pi * f0 * (k + 1) * t) for k, a in harmonics)
    elif kind == "chirp":
        # Quadratic phase → linear f0 → 4·f0 sweep across the clip.
        wave = np.sin(2 * np.pi * (f0 * t + (3 * f0) * t**2 / (2 * seconds)))
    elif kind == "drone":
        f_low = f0 / 2
        wave = (
            np.sin(2 * np.pi * f_low * t)
            + np.sin(2 * np.pi * f_low * 1.01 * t)  # 1% detune → slow beating
            + 0.3 * np.sin(2 * np.pi * f_low * 2.02 * t)
        )
    elif kind == "arpeggio":
        steps = np.array([1.0, 1.25, 1.5, 2.0])  # major triad + octave
        idx = np.minimum((4 * t / seconds).astype(int), 3)
        wave = np.sin(2 * np.pi * f0 * steps[idx] * t)
    elif kind == "percussive":
        wave = rng.standard_normal(t.size)
        wave *= (np.sin(2 * np.pi * 4.0 * t) > 0).astype(np.float64)  # 4 Hz bursts
    else:
        raise ValueError(f"unknown clip kind: {kind!r} (expected one of {CLIP_KINDS})")

    # 10% fade in/out: no clicks at segment boundaries, and the envelope itself
    # is information for pooling (mean/std differ from a constant-amplitude sine).
    envelope = np.minimum(1.0, 10 * t / seconds) * np.minimum(1.0, 10 * (seconds - t) / seconds)
    wave = np.asarray(wave) * envelope
    peak = float(np.max(np.abs(wave))) or 1.0
    return (0.6 * wave / peak).astype(np.float32)


def _write_audio_library(
    dir_path, n: int, sr: int = 48_000, seconds: float = 1.0, seed: int = 0
) -> list[Path]:
    """Write ``n`` distinct WAVs (cycling through :data:`CLIP_KINDS`) and return their paths."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        kind = CLIP_KINDS[i % len(CLIP_KINDS)]
        clip = _synth_clip(kind, seconds=seconds, sr=sr, seed=seed + i)
        path = dir_path / f"{i:02d}_{kind}.wav"
        sf.write(str(path), clip, sr)
        paths.append(path)
    return paths


@pytest.fixture
def synth_clip():
    """Factory: ``synth_clip(kind, seconds=1.0, sr=48_000, seed=0) -> float32 waveform``."""
    return _synth_clip


@pytest.fixture
def make_audio_library():
    """Factory: ``make_audio_library(dir, n, sr=48_000, seconds=1.0) -> list[Path]``."""
    return _write_audio_library


@pytest.fixture
def make_fake_embedder():
    """Factory: ``make_fake_embedder(name, sample_rate, dim=8) -> FakeEmbedder``."""
    return lambda name, sample_rate, dim=8: FakeEmbedder(name, sample_rate, dim)


@pytest.fixture
def fake_clap(make_fake_embedder):
    """A ready-made CLAP-style fake embedder at 48 kHz."""
    return make_fake_embedder("clap", 48_000)
