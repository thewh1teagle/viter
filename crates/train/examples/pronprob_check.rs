//! Estimate lexicon probabilities from a real alignment run and print the four global values,
//! for comparison against MFA's `meta.json`.
//!
//! Usage:
//!   cargo run --release -p viter-train --example pronprob_check -- \
//!       data/lj200-ipa data/ljfull-v8.viter --dict data/dict/ljspeech_ipa_noprobs.dict \
//!       --no-position-dependent

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let corpus_dir = args.first().expect("corpus dir");
    let model_path = args.get(1).expect("model");
    let dict = args
        .iter()
        .position(|a| a == "--dict")
        .and_then(|i| args.get(i + 1))
        .expect("--dict");

    let model = viter_kaldi::model::AcousticModel::load(std::path::Path::new(model_path))?;
    let mut opts = viter_io::corpus::CorpusOptions::default();
    opts.dictionary = Some(std::path::PathBuf::from(dict));
    opts.position_dependent =
        !args.iter().any(|a| a == "--no-position-dependent") && model.position_dependent;
    let mut corpus = viter_io::corpus::scan(std::path::Path::new(corpus_dir), &opts)?;
    viter_io::corpus::remap(&mut corpus, &model.phones)?;

    let device = viter_kaldi::device::Device::auto();
    let results = viter_train::pipeline::align_corpus_with(
        &corpus,
        &model,
        &device,
        None,
        &viter_train::pipeline::AlignOverrides::default(),
    )?;

    let utts: Vec<usize> = (0..corpus.utts.len()).collect();
    let silence = *corpus.silence_phones.first().expect("a silence phone");
    let probs = viter_train::pronprob::estimate_from_intervals(&corpus, silence, &utts, &results);
    println!("silence_prob                  {:.2}", probs.silence_prob);
    println!(
        "initial_silence_prob          {:.2}",
        probs.initial_silence_prob
    );
    println!(
        "final_silence_correction      {:.2}",
        probs.final_silence_correction
    );
    println!(
        "final_non_silence_correction  {:.2}",
        probs.final_non_silence_correction
    );
    Ok(())
}
