# /// script
# requires-python = ">=3.9"
# dependencies = ["viter", "numpy", "soundfile"]
# ///
"""Align audio that is already in memory (a numpy array), no temp files.

    uv run align_numpy.py english.viter english.dict talk.wav "hello world"
"""

import sys

import numpy as np
import soundfile as sf
import viter

model_path, dict_path, audio_path, text = sys.argv[1:5]

samples, sr = sf.read(audio_path, dtype="float32")   # any sample rate; viter resamples
if samples.ndim == 2:
    samples = samples.mean(axis=1)                    # mono

model = viter.Model(model_path, dict=dict_path)
a = model.align(np.asarray(samples), text, sample_rate=sr)

print(f"{len(a.words)} words, {len(a.phones)} phones, {a.duration:.2f} s")
for p in a.phones:
    print(f"{p.start:.3f} {p.end:.3f} {p.label}")
