# Optional final training alignment (#11)

Build the CLI, then run the standalone CPU validation:

```sh
cargo build -p viter
uv run plans/optional_alignment/optional_alignment_001.py
uv run plans/optional_alignment/optional_alignment_001.py --sat
```

`--binary target/release/viter` selects a release build. The default runs monophone
training; `--sat` adds triphone and one SAT round on delta features, exercising the
final two-pass speaker adaptation. A temporary corpus contains six synthetic tone
utterances split across two speaker directories, with no external data dependencies.

Each run trains the same corpus with and without `--out-textgrids` and checks:

- The final stage appears in progress only when exporting TextGrids.
- Skipping it omits alignment counts and prints the `viter align` hint.
- Exporting reports six aligned utterances and zero failures.
- Both runs produce the same stage checkpoints, and each output model is byte-identical
  to its last checkpoint (saved before the optional pass).
- Exported TextGrids mirror the corpus layout and contain words and phones tiers.
- Both saved models subsequently align the full corpus to identical TextGrids, using
  one CPU thread to keep accumulation order fixed.

Model bytes are compared within each process: the phone table contains a HashMap
whose serialized entry order varies across processes. Standalone and training
TextGrids are not compared byte for byte: training's final
pass uses a different silence boost from `viter align`. This validates control flow
and output behavior, not speech accuracy or full-corpus performance.
