import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

__version__: str

#: A path, or anything with ``__fspath__``.
StrPath = Union[str, os.PathLike]
#: A waveform in ``[-1, 1]``: a 1-D float array, or any sequence of floats.
Samples = Union[npt.NDArray[np.float32], npt.NDArray[np.float64], Sequence[float]]
#: Either a file to read, or samples already in memory.
AudioLike = Union[StrPath, Samples]
#: ``(audio, text)`` or ``(audio, text, sample_rate)``.
AlignItem = Union[Tuple[AudioLike, str], Tuple[AudioLike, str, Optional[int]]]

class ViterError(Exception):
    """Raised for every error coming out of viter's Rust core."""

class Phone:
    """One phone interval, in seconds. ``label`` is untagged (``AH_B`` -> ``AH``)."""

    label: str
    start: float
    end: float

class Word:
    """One word interval, with the phones inside it."""

    label: str
    start: float
    end: float
    phones: List[Phone]

class Alignment:
    """A finished alignment of one utterance."""

    words: List[Word]
    phones: List[Phone]
    duration: float
    def to_textgrid(self, path: StrPath) -> None:
        """Write a Praat TextGrid (long format)."""

    def textgrid(self) -> str:
        """The TextGrid text, exactly as :meth:`to_textgrid` would write it."""

    def plot(
        self,
        audio: AudioLike,
        sample_rate: Optional[int] = None,
        *,
        ax: Any = None,
        tiers: Sequence[str] = ("words", "phones"),
        spectrogram: bool = True,
        zoom: Optional[Tuple[float, float]] = None,
        n_mels: int = 80,
        cmap: str = "magma",
    ) -> Any:
        """Plot the alignment over its audio; returns the main matplotlib axes."""

class AlignSummary:
    """What :meth:`Model.align_corpus` did."""

    utterances: int
    aligned: int
    failed: List[str]
    oov_words: Dict[str, int]

class Model:
    """A trained acoustic model, ready to align."""

    def __init__(
        self,
        path: StrPath,
        *,
        dict: Optional[StrPath] = None,
        cpu: bool = False,
    ) -> None: ...
    @property
    def phones(self) -> List[str]: ...
    @property
    def feature_dim(self) -> int: ...
    @property
    def speaker_adapted(self) -> bool: ...
    def save(self, path: StrPath) -> None: ...
    def align(
        self,
        audio: AudioLike,
        text: str,
        *,
        sample_rate: Optional[int] = None,
        beam: Optional[float] = None,
        retry_beam: Optional[float] = None,
        refine: bool = True,
        speaker: Optional[str] = None,
    ) -> Alignment:
        """Align one utterance, raising :class:`ViterError` if it fails."""

    def align_many(
        self,
        items: Sequence[AlignItem],
        *,
        speakers: Optional[Sequence[str]] = None,
        beam: Optional[float] = None,
        retry_beam: Optional[float] = None,
        refine: bool = True,
    ) -> List[Optional[Alignment]]:
        """Align a batch; ``None`` for each utterance that failed."""

    def align_corpus(
        self,
        corpus_dir: StrPath,
        out_dir: StrPath,
        *,
        ctm: bool = False,
        beam: Optional[float] = None,
        retry_beam: Optional[float] = None,
        refine: bool = True,
    ) -> AlignSummary:
        """Align a corpus directory and write TextGrids, like ``viter align``."""

def train(
    corpus_dir: StrPath,
    out: Optional[StrPath] = None,
    *,
    dict: Optional[StrPath] = None,
    config: Union[Mapping[str, Any], StrPath, None] = None,
    cpu: bool = False,
    seed: Optional[int] = None,
    no_tri: bool = False,
    no_lda: bool = False,
    no_sat: bool = False,
    no_pron_probs: bool = False,
    sat_rounds: Optional[int] = None,
    no_subset: bool = False,
    position_dependent: bool = True,
    work_dir: Optional[StrPath] = None,
) -> Model:
    """Train an acoustic model from a corpus directory."""

def import_mfa(path: StrPath, out: Optional[StrPath] = None) -> Model:
    """Convert a Montreal Forced Aligner acoustic model into a viter model."""

def serve(
    dir: StrPath,
    *,
    port: int = 7878,
    open: bool = False,
    audio: Optional[StrPath] = None,
) -> None:
    """Serve the browser viewer; blocks until Ctrl-C."""

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``viter`` CLI in-process and return its exit code."""

def plot(
    alignment: Alignment,
    audio: AudioLike,
    sample_rate: Optional[int] = None,
    *,
    ax: Any = None,
    tiers: Sequence[str] = ("words", "phones"),
    spectrogram: bool = True,
    zoom: Optional[Tuple[float, float]] = None,
    n_mels: int = 80,
    cmap: str = "magma",
) -> Any:
    """Plot an alignment over its audio; needs ``viter[plot]``. Returns the main axes."""
