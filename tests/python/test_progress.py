"""The `progress` callback on `viter.train`: one counter for the whole run."""

import pytest

import viter


def test_progress_callback(lj20):
    corpus, dict_path = lj20
    calls = []

    def on_progress(info):
        calls.append(dict(info))

    viter.train(
        corpus,
        dict=dict_path,
        no_lda=True,
        cpu=True,
        progress=on_progress,
        quiet=True,
    )

    assert calls, "the callback was never called"

    keys = {
        "done",
        "total",
        "fraction",
        "elapsed",
        "eta",
        "stage",
        "step",
        "mismatches",
    }
    assert keys <= set(calls[0]), f"missing keys: {keys - set(calls[0])}"

    dones = [c["done"] for c in calls]
    assert dones == sorted(dones), "done went backwards"

    fractions = [c["fraction"] for c in calls]
    assert fractions == sorted(fractions), "fraction went backwards"
    assert all(0.0 <= f <= 1.0 for f in fractions), "fraction left [0, 1]"
    assert fractions[-1] == 1.0, "the bar did not end at 100%"

    elapsed = [c["elapsed"] for c in calls]
    assert elapsed == sorted(elapsed), "elapsed went backwards"

    assert all(
        c["eta"] is None or c["eta"] >= 0.0 for c in calls
    ), "a negative ETA was reported"

    total = calls[-1]["total"]
    assert total > 0
    assert calls[-1]["done"] == total, "the run did not finish at 100%"
    assert all(c["total"] == total for c in calls), "total changed mid-run"
    assert all(c["stage"] for c in calls[1:]), "stage was empty after the first call"
    assert calls[-1]["mismatches"] == 0, "the run disagreed with the plan"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_progress_callback_exceptions_are_swallowed(lj20):
    """A raising callback must not abort a run that may be many minutes in."""
    corpus, dict_path = lj20
    seen = []

    def on_progress(info):
        seen.append(info["done"])
        raise RuntimeError("callback blew up")

    model = viter.train(
        corpus,
        dict=dict_path,
        no_tri=True,
        cpu=True,
        progress=on_progress,
        quiet=True,
    )
    assert model is not None
    assert seen, "the callback was never called"
