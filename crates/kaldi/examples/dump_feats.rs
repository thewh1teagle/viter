//! Scratch parity harness: dump viter's feature pipeline stages for one wav.
use std::io::Write;
use std::path::Path;

use viter_kaldi::feat::{
    CmvnStats, DeltaOptions, MfccComputer, MfccOptions, add_deltas, apply_cmvn,
};
use viter_kaldi::types::Feats;

fn dump(path: &str, f: &Feats) {
    let mut out = std::fs::File::create(path).unwrap();
    out.write_all(&(f.nrows() as u32).to_le_bytes()).unwrap();
    out.write_all(&(f.ncols() as u32).to_le_bytes()).unwrap();
    for v in f.iter() {
        out.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let wav = &args[1];
    let prefix = &args[2];

    let audio = viter_kaldi::audio::read_16k(Path::new(wav)).unwrap();
    {
        let mut out = std::fs::File::create(format!("{prefix}.wave.bin")).unwrap();
        for s in &audio.samples {
            out.write_all(&s.to_le_bytes()).unwrap();
        }
    }

    let computer = MfccComputer::new(MfccOptions::default());
    let raw = computer.compute(&audio.samples);
    dump(&format!("{prefix}.raw.bin"), &raw);

    let mut stats = CmvnStats::new(raw.ncols());
    stats.accumulate(&raw);
    let mut cmvned = raw.clone();
    apply_cmvn(&mut cmvned, &stats, false);
    dump(&format!("{prefix}.cmvn.bin"), &cmvned);

    let d = add_deltas(&cmvned, &DeltaOptions::default());
    dump(&format!("{prefix}.deltas.bin"), &d);
    eprintln!(
        "frames={} dim={} deltas_dim={}",
        raw.nrows(),
        raw.ncols(),
        d.ncols()
    );
}
