"""Notebook visualisation of an alignment, in the spirit of ``librosa.display.specshow``.

Pure Python and numpy; matplotlib is imported lazily, so this module stays importable
without the ``plot`` extra installed.
"""

import wave

import numpy as np

__all__ = ["plot"]

_WIN = 0.025
_HOP = 0.010


def _read_wav(path):
    """Read a wav file with the stdlib, returning ``(mono float array, sample_rate)``."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        width = w.getsampwidth()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width == 1:  # 8-bit wav is unsigned
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav sample width: {width} bytes")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def _hz_to_mel(f):
    """Slaney mel: linear below 1 kHz, log above."""
    f = np.asarray(f, dtype=np.float64)
    lin = f / (200.0 / 3.0)
    log = 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f < 1000.0, lin, log)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    lin = m * (200.0 / 3.0)
    log = 1000.0 * np.exp((m - 15.0) * (np.log(6.4) / 27.0))
    return np.where(m < 15.0, lin, log)


def _mel_filters(n_mels, n_fft, sr):
    """Slaney-style (area-normalised) triangular mel filterbank."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2.0), n_mels + 2))
    fb = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (freqs - lo) / max(mid - lo, 1e-9)
        right = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(left, right)) * (2.0 / max(hi - lo, 1e-9))
    return fb


def _logmel(y, sr, n_mels):
    """A dB-scaled log-mel spectrogram, plus the frame times."""
    win_len = max(16, int(round(_WIN * sr)))
    hop = max(1, int(round(_HOP * sr)))
    n_fft = 1 << (win_len - 1).bit_length()
    window = np.hanning(win_len + 1)[:win_len]
    pad = np.pad(y, (0, max(0, win_len - y.size)))
    n_frames = max(1, 1 + (pad.size - win_len) // hop)
    idx = np.arange(win_len)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = pad[idx] * window
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    mel = spec @ _mel_filters(n_mels, n_fft, sr).T
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db = np.maximum(db, db.max() - 80.0)
    times = (np.arange(n_frames) * hop + win_len / 2.0) / sr
    return db.T, times


def _intervals(alignment, tier):
    if tier in ("words", "word"):
        items = alignment.words
    elif tier in ("phones", "phone"):
        items = alignment.phones
    else:
        raise ValueError(f"unknown tier: {tier!r} (expected 'words' or 'phones')")
    return [(i.start, i.end, i.label) for i in items]


def _draw_tier(ax, intervals, name, xlim, rectangle):
    """One strip of labelled interval rectangles."""
    ax.set_ylim(0, 1)
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    span = max(xlim[1] - xlim[0], 1e-9)
    fig = ax.figure
    axes_pt = fig.get_size_inches()[0] * 72.0 * ax.get_position().width
    for start, end, label in intervals:
        if end <= xlim[0] or start >= xlim[1]:
            continue
        hollow = not label.strip() or label.strip() in ("sil", "sp", "spn", "<eps>")
        face = "none" if hollow else "#3b6ea5"
        rect = rectangle(
            (start, 0.05), end - start, 0.9,
            facecolor=face, edgecolor="#3b6ea5",
            alpha=0.6 if hollow else 0.35, linewidth=0.8,
        )
        ax.add_patch(rect)
        if hollow:
            continue
        # Fit the label: box width in points (from the real axes width), then a
        # font size at which ~0.6 em per character fits, capped at 9 pt; below 4 pt
        # the text is unreadable and skipped rather than smeared.
        frac = (min(end, xlim[1]) - max(start, xlim[0])) / span
        box_pt = frac * axes_pt
        size = min(9.0, box_pt / (0.6 * len(label) + 1.0))
        if size < 4.0:
            continue
        mid = 0.5 * (start + end)
        ax.text(mid, 0.5, label, ha="center", va="center", fontsize=size, clip_on=True)


def plot(
    alignment,
    audio,
    sample_rate=None,
    *,
    ax=None,
    tiers=("words", "phones"),
    spectrogram=True,
    zoom=None,
    n_mels=80,
    cmap="magma",
):
    """Plot ``alignment`` over ``audio`` and return the main (spectrogram) axes."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("matplotlib is required: pip install viter[plot]") from exc

    if isinstance(audio, str) or hasattr(audio, "__fspath__"):
        y, sr = _read_wav(audio)
        y = y.astype(np.float64)
    else:
        y = np.asarray(audio, dtype=np.float64).reshape(-1)
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is an array")
        sr = int(sample_rate)
    if y.size == 0:
        raise ValueError("audio is empty")

    tiers = tuple(tiers)
    xlim = tuple(zoom) if zoom else (0.0, max(alignment.duration, y.size / sr))

    if ax is None:
        _, axes = plt.subplots(
            len(tiers) + 1,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [4] + [1] * len(tiers), "hspace": 0.05},
            figsize=(12, 3 + 0.6 * len(tiers)),
        )
        axes = np.atleast_1d(axes)
        main, strips = axes[0], list(axes[1:])
    else:
        main = ax
        n = len(tiers)
        strips = [
            main.inset_axes([0.0, -0.16 * (i + 1), 1.0, 0.14], transform=main.transAxes)
            for i in range(n)
        ]

    if spectrogram:
        db, times = _logmel(y, sr, n_mels)
        main.imshow(
            db,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=(float(times[0]), float(times[-1]), 0.0, n_mels),
        )
        main.set_ylabel("mel")
    else:
        t = np.arange(y.size) / sr
        main.plot(t, y, linewidth=0.5, color="#3b6ea5")
        main.set_ylabel("amplitude")
    main.set_xlim(*xlim)

    # Boundaries from the finest requested tier, through the spectrogram too.
    line_c = "w" if spectrogram else "0.5"
    finest = _intervals(alignment, tiers[-1]) if tiers else []
    for start, end, _ in finest:
        for x in (start, end):
            if xlim[0] <= x <= xlim[1]:
                main.axvline(x, color=line_c, linewidth=0.5, alpha=0.6)

    for strip, tier in zip(strips, tiers):
        _draw_tier(strip, _intervals(alignment, tier), tier, xlim, Rectangle)
        strip.set_xlim(*xlim)
    (strips[-1] if strips else main).set_xlabel("time (s)")
    for strip in strips[:-1]:
        strip.set_xticklabels([])
    return main
