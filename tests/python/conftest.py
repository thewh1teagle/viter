"""Shared paths for the python tests."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LJ20 = REPO / "tests" / "data" / "lj20"
CMUDICT = REPO / "tests" / "data" / "cmudict.dict"


@pytest.fixture(scope="session")
def lj20():
    """The 20-utterance LJSpeech sample and cmudict, or skip (both are gitignored)."""
    if not LJ20.is_dir() or not CMUDICT.is_file():
        pytest.skip("tests/data/lj20 or tests/data/cmudict.dict missing")
    return LJ20, CMUDICT
