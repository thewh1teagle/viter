# /// script
# requires-python = ">=3.9"
# dependencies = ["viter"]
# ///
"""Align many utterances in one call, grouped by speaker for fMLLR adaptation.

    uv run align_many.py english.viter english.dict clips/
"""

import sys
from pathlib import Path

import viter

model_path, dict_path, clips = sys.argv[1:4]

items, speakers = [], []
for wav in sorted(Path(clips).glob("**/*.wav")):
    txt = wav.with_suffix(".txt")
    if not txt.exists():
        continue
    items.append((str(wav), txt.read_text().strip()))
    speakers.append(wav.parent.name)          # one folder per speaker

model = viter.Model(model_path, dict=dict_path)
results = model.align_many(items, speakers=speakers)   # runs across CPU cores / GPU, GIL released

failed = 0
for (wav, _), a in zip(items, results):
    if a is None:
        failed += 1
        print(f"FAILED {wav}")
        continue
    a.to_textgrid(Path("aligned") / (Path(wav).stem + ".TextGrid"))

print(f"aligned {len(items) - failed}/{len(items)}")
