# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compare full-schedule training on a bounded LJSpeech slice, never the full corpus."""
import argparse
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("data/ljfull"))
    parser.add_argument("--dict", type=Path, default=Path("data/dict/ljspeech_ipa_noprobs.dict"))
    parser.add_argument("--utts", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("/tmp/viter-small-training"))
    args = parser.parse_args()
    assert 1 <= args.utts <= 128, "training validation is limited to 128 utterances"
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    corpus = root / "corpus" / "LJ"
    corpus.mkdir(parents=True)
    sources = sorted(args.corpus.rglob("*.wav"))
    assert len(sources) > args.utts, "this script must not train on the full source corpus"
    for i in range(args.utts):
        wav = sources[i * len(sources) // args.utts]
        text = wav.with_suffix(".txt")
        assert text.is_file(), text
        (corpus / wav.name).symlink_to(wav.resolve())
        (corpus / text.name).symlink_to(text.resolve())
    timings = {}
    for name, binary in (("baseline", args.baseline), ("candidate", args.candidate)):
        start = time.perf_counter()
        with (root / f"{name}.log").open("w") as log:
            result = subprocess.run([
                str(binary.resolve()), "train", str(corpus.parent),
                "--dict", str(args.dict.resolve()), "--no-position-dependent",
                "--seed", "0", "-o", str(root / f"{name}.viter"),
                "--out-textgrids", str(root / name),
                "--work-dir", str(root / f"{name}-stages"),
            ], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "RUST_LOG": "error", "RAYON_NUM_THREADS": "20"}, timeout=180)
        assert result.returncode == 0, f"{name} failed; see {root / (name + '.log')}"
        timings[name] = time.perf_counter() - start
        print(f"{name}: {timings[name]:.3f}s", flush=True)
    baseline = {p.relative_to(root / "baseline"): p for p in (root / "baseline").rglob("*.TextGrid")}
    candidate = {p.relative_to(root / "candidate"): p for p in (root / "candidate").rglob("*.TextGrid")}
    assert len(baseline) == args.utts, f"baseline aligned {len(baseline)}/{args.utts}"
    assert baseline.keys() == candidate.keys(), "alignment successes differ"
    different = [str(p) for p in baseline if baseline[p].read_bytes() != candidate[p].read_bytes()]
    assert not different, f"TextGrid outputs differ: {different}"
    print(f"PASS: {args.utts} TextGrids byte-identical; full schedule on bounded slice; "
          f"speedup={timings['baseline'] / timings['candidate']:.3f}x; logs={root}")


if __name__ == "__main__":
    main()
