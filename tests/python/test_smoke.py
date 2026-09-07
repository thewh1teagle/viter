"""End-to-end smoke test on a synthetic phoneme-mode corpus. CPU only, monophone only."""

import math
import struct
import wave

import numpy as np
import pytest

import viter

SR = 16000
# Three pseudo-phones, each a distinct steady tone, so the monophone GMMs have
# something separable to fit.
TONES = {"a": 220.0, "b": 660.0, "c": 1320.0}
UTTS = ["a b a", "b a b", "a c a", "c b c", "b c a", "a b c"]


def synth(text: str) -> np.ndarray:
    """One ~1 s waveform: a tone per token, with a little noise and short silences."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    parts = [np.zeros(int(0.05 * SR), dtype=np.float32)]
    tokens = text.split()
    dur = 0.9 / len(tokens)
    t = np.arange(int(dur * SR), dtype=np.float32) / SR
    for tok in tokens:
        wave_ = 0.4 * np.sin(2 * math.pi * TONES[tok] * t)
        parts.append((wave_ + 0.01 * rng.standard_normal(t.size)).astype(np.float32))
    parts.append(np.zeros(int(0.05 * SR), dtype=np.float32))
    return np.clip(np.concatenate(parts), -1.0, 1.0)


def write_wav(path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(struct.pack(f"<{pcm.size}h", *pcm.tolist()))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    for i, text in enumerate(UTTS):
        write_wav(d / f"u{i}.wav", synth(text))
        (d / f"u{i}.txt").write_text(text + "\n")
    return d


@pytest.fixture(scope="module")
def model(corpus):
    return viter.train(corpus, no_tri=True, cpu=True)


def test_version():
    assert isinstance(viter.__version__, str)
    assert viter.__version__


def test_cli_help(capfd):
    assert viter.main(["viter", "--help"]) == 0


def test_model_attributes(model):
    assert isinstance(model.phones, list)
    # phones carry the position tags of the model (a_S, ...); alignment labels do not.
    assert set(TONES) <= {p.split("_")[0] for p in model.phones}
    assert model.feature_dim > 0
    assert isinstance(model.speaker_adapted, bool)


def test_align_path_and_array_agree(corpus, model, tmp_path):
    text = UTTS[0]
    wav = corpus / "u0.wav"

    from_path = model.align(wav, text)
    samples = synth(text)  # deterministic: same seed, same waveform as the file
    from_array = model.align(samples, text, sample_rate=SR)

    words = [w.label for w in from_path.words]
    assert words == [w.label for w in from_array.words]
    assert [w for w in words if w in TONES] == text.split()

    assert from_path.duration > 0
    for w in from_path.words:
        assert 0.0 <= w.start <= w.end <= from_path.duration + 1e-6
    for p in from_path.phones:
        assert 0.0 <= p.start <= p.end <= from_path.duration + 1e-6
    assert from_path.phones

    tg = tmp_path / "u0.TextGrid"
    from_path.to_textgrid(tg)
    assert tg.read_text().startswith('File type = "ooTextFile"')
    assert from_path.textgrid().startswith('File type = "ooTextFile"')


def test_align_many(corpus, model):
    items = [(corpus / f"u{i}.wav", UTTS[i]) for i in range(3)]
    out = model.align_many(items)
    assert len(out) == 3
    for a, text in zip(out, UTTS):
        assert a is not None
        assert [w.label for w in a.words if w.label in TONES] == text.split()


def test_save_and_reload(model, corpus, tmp_path):
    path = tmp_path / "m.viter"
    model.save(path)
    again = viter.Model(path, cpu=True)
    assert again.phones == model.phones


def test_align_corpus(corpus, model, tmp_path):
    out = tmp_path / "aligned"
    summary = model.align_corpus(corpus, out)
    assert summary.utterances == len(UTTS)
    assert summary.aligned >= 1
    assert list(out.glob("**/*.TextGrid"))


def test_error_type():
    assert issubclass(viter.ViterError, Exception)
    with pytest.raises(viter.ViterError):
        viter.Model("no/such/model.viter")
