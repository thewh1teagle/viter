"""The optional matplotlib visualisation, over the same synthetic corpus as the smoke test."""

import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import viter  # noqa: E402

from test_smoke import SR, UTTS, synth, write_wav  # noqa: E402


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("plot_corpus")
    for i, text in enumerate(UTTS):
        write_wav(d / f"u{i}.wav", synth(text))
        (d / f"u{i}.txt").write_text(text + "\n")
    return d


@pytest.fixture(scope="module")
def aligned(corpus):
    model = viter.train(corpus, no_tri=True, cpu=True)
    return model.align(corpus / "u0.wav", UTTS[0]), corpus / "u0.wav"


def test_plot_from_array(aligned, tmp_path):
    a, _ = aligned
    ax = a.plot(synth(UTTS[0]), sample_rate=SR)
    out = tmp_path / "array.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 0


def test_plot_from_wav_waveform(aligned, tmp_path):
    a, wav = aligned
    ax = viter.plot(a, wav, spectrogram=False, tiers=("phones",), zoom=(0, 0.5))
    out = tmp_path / "wave.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 0
