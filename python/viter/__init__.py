"""viter — a rusty forced aligner. Train, align, serve.

Everything here is implemented in Rust and exposed through the ``viter._viter``
extension module; this file only re-exports it so ``import viter`` is enough.
"""

from ._viter import (
    Alignment,
    AlignSummary,
    Model,
    Phone,
    ViterError,
    Word,
    __version__,
    import_mfa,
    main,
    serve,
    train,
)
from .plot import plot

__all__ = [
    "Alignment",
    "AlignSummary",
    "Model",
    "Phone",
    "ViterError",
    "Word",
    "__version__",
    "import_mfa",
    "main",
    "plot",
    "serve",
    "train",
]
