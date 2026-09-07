# /// script
# requires-python = ">=3.11"
# ///
"""Compare bounded frozen-model replay timing and complete numerical artifacts."""
import argparse
import array
import math
import pathlib
import re
import statistics


def phases(path):
    samples = {}
    for line in pathlib.Path(path).read_text().splitlines():
        match = re.fullmatch(r"phase=(\w+) repeat=(\d+) ms=([\d.]+)", line)
        if match and int(match[2]) > 0:
            samples.setdefault(match[1], []).append(float(match[3]))
    return {key: statistics.median(values) for key, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="artifact prefix, with .log/.align/.stats")
    parser.add_argument("candidate", help="artifact prefix, with .log/.align/.stats")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    before, after = pathlib.Path(args.baseline), pathlib.Path(args.candidate)
    aligned = before.with_suffix(".align").read_bytes() == after.with_suffix(".align").read_bytes()
    print(f"Exact transition/word/pronunciation paths: {aligned}")
    a, b = array.array("d"), array.array("d")
    a.frombytes(before.with_suffix(".stats").read_bytes())
    b.frombytes(after.with_suffix(".stats").read_bytes())
    assert len(a) == len(b), "Statistic shapes differ"
    failures = sum(not (math.isfinite(x) and math.isfinite(y) and abs(x-y) <= args.atol + args.rtol*abs(x)) for x, y in zip(a, b))
    maximum = max((abs(x-y) for x,y in zip(a,b)), default=0)
    print(f"Statistics: {len(a)} values, {failures} outside tolerance; max absolute delta {maximum:.9g}")
    t0, t1 = phases(before.with_suffix(".log")), phases(after.with_suffix(".log"))
    for name in t0.keys() & t1.keys():
        print(f"{name:20s} {t0[name]:9.3f} -> {t1[name]:9.3f} ms ({t0[name]/t1[name]:.3f}x)")
    assert aligned, "Alignment paths changed"
    assert not failures, "Sufficient statistics changed beyond supplied tolerance"
    extra0, extra1 = before.with_suffix(".extra"), after.with_suffix(".extra")
    assert extra0.exists() == extra1.exists(), "Only one replay includes fMLLR artifacts"
    if extra0.exists():
        a, b = array.array("d"), array.array("d")
        a.frombytes(extra0.read_bytes())
        b.frombytes(extra1.read_bytes())
        assert len(a) == len(b), "fMLLR shapes differ"
        errors = sum(not (math.isfinite(x) and math.isfinite(y) and abs(x-y) <= args.atol + args.rtol*abs(x)) for x,y in zip(a,b))
        maximum = max((abs(x-y) for x,y in zip(a,b)), default=0)
        print(f"fMLLR statistics/transforms: {len(a)} values, {errors} outside tolerance; max absolute delta {maximum:.9g}")
        assert not errors, "fMLLR statistics/transforms changed beyond supplied tolerance"


if __name__ == "__main__":
    main()
