#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Check optional training alignment through the CLI on a synthetic corpus."""

import argparse
import math
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import tempfile
import wave


def make_corpus(root: Path) -> int:
    tones = {"a": 220, "b": 660, "c": 1320}
    texts = ["a b a", "b a b", "a c a", "c b c", "b c a", "a b c"]
    rng = random.Random(11)
    sr = 16000
    for i, text in enumerate(texts):
        stem = root / f"speaker{i % 2}" / f"u{i}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        samples = [0] * (sr // 20)
        for phone in text.split():
            samples.extend(
                int(32767 * (0.4 * math.sin(2 * math.pi * tones[phone] * j / sr)
                             + rng.uniform(-0.01, 0.01)))
                for j in range(4800)
            )
        samples.extend([0] * (sr // 20))
        with wave.open(str(stem.with_suffix(".wav")), "wb") as wav:
            wav.setparams((1, 2, sr, 0, "NONE", "not compressed"))
            wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        stem.with_suffix(".txt").write_text(text + "\n")
    return len(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/viter"))
    parser.add_argument("--sat", action="store_true", help="include triphone and one SAT round")
    args = parser.parse_args()
    binary = args.binary.resolve()

    def run(*argv: object) -> str:
        result = subprocess.run(
            [str(binary), *map(str, argv)], capture_output=True, text=True,
            env={**os.environ, "RAYON_NUM_THREADS": "1", "NO_COLOR": "1"},
            timeout=300,
        )
        output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
        assert result.returncode == 0, output
        return output

    with tempfile.TemporaryDirectory(prefix="viter-optional-alignment-") as tmp:
        root = Path(tmp)
        corpus = root / "corpus"
        count = make_corpus(corpus)
        flags = ["--no-lda", "--sat-rounds", "1"] if args.sat else ["--no-tri"]
        common = ["train", corpus, "--cpu", "--no-pron-probs", *flags]
        logs = {}
        for name, export in (("skipped", False), ("exported", True)):
            extra = ["--out-textgrids", root / "textgrids"] if export else []
            logs[name] = run(
                *common, "-o", root / f"{name}.viter", "--work-dir", root / name, *extra,
            )
            assert (root / f"{name}.viter").is_file()
            plan = next(line for line in logs[name].splitlines() if "Plan" in line)
            assert ("Aligning" in plan) == export, plan
            assert ("utterances aligned" in logs[name]) == export, logs[name]
            for field in ("aligned", "failed"):
                assert bool(re.search(rf"^\s*{field}:\s+\d+", logs[name], re.M)) == export, logs[name]

        assert "run `viter align`" in logs["skipped"]
        assert re.search(rf"^\s*aligned:\s+{count}\b", logs["exported"], re.M)
        assert re.search(r"^\s*failed:\s+0\b", logs["exported"], re.M)
        checkpoints = sorted(p.name for p in (root / "skipped").glob("*.viter"))
        assert checkpoints
        assert checkpoints == sorted(p.name for p in (root / "exported").glob("*.viter"))
        last = "sat.viter" if args.sat else "mono.viter"
        for name in ("skipped", "exported"):
            # Within one process the symbol table's HashMap keeps its serialization
            # order. Compare the output with the checkpoint saved BEFORE the final pass.
            assert (root / f"{name}.viter").read_bytes() == (root / name / last).read_bytes()

        for name in ("skipped", "exported"):
            run("align", corpus, root / f"{name}.viter", "--cpu", "--no-refine",
                "-o", root / f"later-{name}")
        expected = {Path(f"speaker{i % 2}/u{i}.TextGrid") for i in range(count)}
        for folder in (root / "textgrids", root / "later-skipped", root / "later-exported"):
            grids = {p.relative_to(folder) for p in folder.rglob("*.TextGrid")}
            assert grids == expected, grids
            for path in grids:
                text = (folder / path).read_text()
                assert 'name = "words"' in text and 'name = "phones"' in text
        for path in expected:
            assert (root / "later-skipped" / path).read_bytes() == (root / "later-exported" / path).read_bytes()
        print(f"PASS: {count} utterances; optional progress/counts, unchanged final "
              "checkpoint models, exported TextGrids, and identical standalone alignments")


if __name__ == "__main__":
    main()
