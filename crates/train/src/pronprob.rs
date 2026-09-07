//! Pronunciation and silence probability estimation from alignments.
//!
//! A port of MFA's `pronunciation_probabilities` stage: counting, from the training
//! alignments, how often each pronunciation of each word was used, how often silence
//! follows it, and how often silence precedes it relative to what the global silence
//! rate would predict. The four resulting numbers per pronunciation are exactly the
//! four probability columns of an MFA probabilistic dictionary and map one-to-one onto
//! [`Pronunciation`]'s optional fields, which `hmm::graph` already turns into arc costs.
//!
//! Sources (formulas replicated line for line):
//! - `plans/mfa/montreal_forced_aligner/alignment/multiprocessing.py:1465-1500`
//!   (`GeneratePronunciationsFunction._process_pronunciations`): the counting pass.
//! - `plans/mfa/montreal_forced_aligner/alignment/base.py:307-535`
//!   (`compute_pronunciation_probabilities`): smoothing and the final formulas.
//! - `plans/mfa/montreal_forced_aligner/data.py:2007-2060`
//!   (`PronunciationProbabilityCounter`): the counter fields.
//! - `plans/mfa/montreal_forced_aligner/helper.py:571-581`
//!   (`format_probability`, `format_correction`): rounding and clamping.

use std::collections::HashMap;

use viter_io::corpus::Corpus;
use viter_kaldi::hmm::{self, TransitionModel};
use viter_kaldi::types::{Alignment, IntervalAlignment, PhoneId, Pronunciation};

pub use viter_kaldi::model::{LexiconProbs, PronProb};

/// MFA's `lambda_2` (`base.py:348`): smoothing weight for the silence-after probability.
const LAMBDA_2: f32 = 2.0;
/// MFA's `lambda_3` (`base.py:452`): smoothing weight for the before-corrections.
const LAMBDA_3: f32 = 2.0;

/// `format_probability` (`helper.py:571-573`): round to two decimals, clamp to [0.01, 0.99].
fn format_probability(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

/// As above, with MFA's `min(max(.., 0.01), 0.99)` clamp.
fn clamp_probability(v: f32) -> f32 {
    format_probability(v).clamp(0.01, 0.99)
}

/// `format_correction` (`helper.py:576-581`): round to two decimals, floor at 0.01.
fn format_correction(v: f32) -> f32 {
    let v = (v * 100.0).round() / 100.0;
    if v <= 0.0 { 0.01 } else { v }
}

/// One word token of an utterance as MFA's counter sees it: the word string and which
/// pronunciation of it was aligned. MFA keys on `(word, pronunciation string)`; we key on
/// `(word, pron index)`, which is the same partition of the data.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Token {
    /// MFA's `("<s>", "")` sentinel (`multiprocessing.py:1478`).
    Start,
    /// MFA's `("</s>", "")` sentinel.
    End,
    Word {
        word: u32,
        pron: u32,
    },
}

/// Counts collected over the whole corpus, mirroring `PronunciationProbabilityCounter`
/// (`data.py:2007-2060`). The word key of a `Token::Word` indexes `words`, the interned
/// word-string table built alongside these counts.
#[derive(Default)]
struct Counter {
    /// `word_pronunciation_counts[word][pron]`.
    word_pron_counts: HashMap<u32, HashMap<u32, f32>>,
    silence_following: HashMap<Token, f32>,
    non_silence_following: HashMap<Token, f32>,
    silence_before: HashMap<Token, f32>,
    non_silence_before: HashMap<Token, f32>,
    /// `ngram_counts[(w_p1, w_p2)] = (silence, non_silence)`.
    ngram: HashMap<(Token, Token), (f32, f32)>,
}

fn bump(m: &mut HashMap<Token, f32>, k: Token) {
    *m.entry(k).or_insert(0.0) += 1.0;
}

impl Counter {
    /// `_process_pronunciations` (`multiprocessing.py:1465-1500`), one utterance.
    ///
    /// `seq` is the utterance's non-silence word tokens in order; `sil_after[i]` says whether
    /// an optional silence was aligned between token `i` and token `i+1` (and `sil_after` has
    /// one extra trailing entry for silence before `</s>`, plus a leading entry, handled by
    /// the caller passing `seq`/`sil_between` already framed by the sentinels).
    fn add_utterance(&mut self, seq: &[Token], sil_between: &[bool]) {
        debug_assert_eq!(sil_between.len() + 1, seq.len());
        for (i, &w_p) in seq.iter().enumerate() {
            // "if i != 0: silence_before/non_silence_before" — the gap *before* token i.
            if i != 0 {
                if sil_between[i - 1] {
                    bump(&mut self.silence_before, w_p);
                } else {
                    bump(&mut self.non_silence_before, w_p);
                }
            }
            // MFA skips silence words entirely; our `seq` never contains them.
            if let Token::Word { word, pron } = w_p {
                *self
                    .word_pron_counts
                    .entry(word)
                    .or_default()
                    .entry(pron)
                    .or_insert(0.0) += 1.0;
            }
            if matches!(w_p, Token::End) {
                continue;
            }
            // The gap *after* token i, and the bigram it starts.
            if i + 1 < seq.len() {
                if sil_between[i] {
                    bump(&mut self.silence_following, w_p);
                    // MFA's ngram key skips the silence word: (w_p, the token after the silence).
                    if i + 1 < seq.len() {
                        let e = self.ngram.entry((w_p, seq[i + 1])).or_insert((0.0, 0.0));
                        e.0 += 1.0;
                    }
                } else {
                    bump(&mut self.non_silence_following, w_p);
                    let e = self.ngram.entry((w_p, seq[i + 1])).or_insert((0.0, 0.0));
                    e.1 += 1.0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Estimate lexicon probabilities from frame-level alignments (MFA's algorithm).
///
/// `alignments[i]` is the alignment of `corpus.utts[utts[i]]`; `None` entries (failed
/// alignments) are skipped, as MFA skips utterances with no alignment.
pub fn estimate(
    corpus: &Corpus,
    tm: &TransitionModel,
    silence_phone: PhoneId,
    utts: &[usize],
    alignments: &[Option<Alignment>],
) -> LexiconProbs {
    let intervals: Vec<Option<IntervalAlignment>> = alignments
        .iter()
        .zip(utts)
        .map(|(a, &u)| {
            a.as_ref()
                .map(|a| hmm::to_intervals(tm, a, &corpus.utts[u].prons, 0.01))
        })
        .collect();
    estimate_from_intervals(corpus, silence_phone, utts, &intervals)
}

/// As [`estimate`], but from alignments already converted to intervals — the form the
/// aligner writes to TextGrids, and the form the tests use.
pub fn estimate_from_intervals(
    corpus: &Corpus,
    silence_phone: PhoneId,
    utts: &[usize],
    intervals: &[Option<IntervalAlignment>],
) -> LexiconProbs {
    // Intern word strings so the counter can key on small ids.
    let mut word_ids: HashMap<&str, u32> = HashMap::new();
    let mut word_strs: Vec<&str> = Vec::new();
    let mut counter = Counter::default();

    for (slot, &u) in intervals.iter().zip(utts) {
        let Some(ia) = slot else { continue };
        let utt = &corpus.utts[u];
        let (seq, sil) = utterance_tokens(utt, ia, silence_phone, &mut word_ids, &mut word_strs);
        counter.add_utterance(&seq, &sil);
    }

    finalize(corpus, &counter, &word_strs, &word_ids)
}

/// Override a word's pronunciation probability fields with the learned values.
///
/// Words with no learned entry, and pronunciation indices beyond what was learned, keep
/// whatever the lexicon gave them (typically `None`, i.e. the graph's defaults).
pub fn apply(probs: &LexiconProbs, word: &str, prons: &[Pronunciation]) -> Vec<Pronunciation> {
    let learned = probs.words.get(word);
    prons
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mut p = p.clone();
            if let Some(pp) = learned.and_then(|v| v.get(i)) {
                p.prob = Some(pp.prob);
                p.silence_after_prob = Some(pp.silence_after_prob);
                p.silence_before_correction = Some(pp.silence_before_correction);
                p.non_silence_before_correction = Some(pp.non_silence_before_correction);
            }
            p
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Counting
// ---------------------------------------------------------------------------

/// Turn one aligned utterance into MFA's `[("<s>","")] + word_pronunciations + [("</s>","")]`
/// sequence plus the between-token silence flags.
///
/// A gap carries silence when a silence-phone interval sits between the two words' intervals
/// (`hmm::to_intervals` leaves inserted optional silence outside every word interval).
fn utterance_tokens<'c>(
    utt: &'c viter_kaldi::types::Utterance,
    ia: &IntervalAlignment,
    silence_phone: PhoneId,
    word_ids: &mut HashMap<&'c str, u32>,
    word_strs: &mut Vec<&'c str>,
) -> (Vec<Token>, Vec<bool>) {
    let mut seq = vec![Token::Start];
    let mut sil = Vec::new();
    let mut prev_end = 0u32;

    let sil_in = |from: u32, to: u32| {
        ia.phones
            .iter()
            .any(|p| p.phone == silence_phone && p.start_frame >= from && p.end_frame <= to)
    };

    for wi in &ia.words {
        let Some(text) = utt.words.get(wi.word as usize) else {
            continue;
        };
        // Silence words themselves are never tokens (MFA: `silence_check`); they only
        // appear as the silence that separates two word tokens, which `to_intervals`
        // already leaves outside word intervals.
        let next = *word_ids.entry(text.as_str()).or_insert_with(|| {
            word_strs.push(text.as_str());
            (word_strs.len() - 1) as u32
        });
        sil.push(sil_in(prev_end, wi.start_frame));
        seq.push(Token::Word {
            word: next,
            pron: wi.pron,
        });
        prev_end = wi.end_frame;
    }

    // The final gap, before `</s>`.
    let last = ia.phones.last().map(|p| p.end_frame).unwrap_or(prev_end);
    sil.push(sil_in(prev_end, last));
    seq.push(Token::End);
    (seq, sil)
}

// ---------------------------------------------------------------------------
// The formulas (base.py:365-535)
// ---------------------------------------------------------------------------

fn finalize(
    corpus: &Corpus,
    counter: &Counter,
    word_strs: &[&str],
    word_ids: &HashMap<&str, u32>,
) -> LexiconProbs {
    // base.py:369-372: the global silence probability over every observed word boundary.
    let silence_count: f32 = counter.silence_before.values().sum();
    let non_silence_count: f32 = counter.non_silence_before.values().sum();
    let total = silence_count + non_silence_count;
    if total == 0.0 {
        return LexiconProbs {
            words: HashMap::new(),
            silence_prob: 0.5,
            initial_silence_prob: 0.5,
            final_silence_correction: 1.0,
            final_non_silence_correction: 1.0,
        };
    }
    // base.py:435: `silence_probability = format_probability(silence_count / total)`.
    let silence_probability = clamp_probability(silence_count / total);

    // Every (word, pron) the lexicon lists for a word that was actually seen: MFA's
    // `pronunciations` query, restricted to `Word.count > 0`.
    let mut all_prons: Vec<(u32, u32)> = Vec::new();
    let mut lex_counts: HashMap<u32, HashMap<u32, f32>> = HashMap::new();
    for utt in &corpus.utts {
        for (w, prons) in utt.words.iter().zip(&utt.prons) {
            let Some(&wid) = word_ids.get(w.as_str()) else {
                continue;
            };
            if lex_counts.contains_key(&wid) {
                continue;
            }
            let e = lex_counts.entry(wid).or_default();
            for pi in 0..prons.len() as u32 {
                // base.py:419: "Add one smoothing" — every listed pronunciation starts at 1,
                // on top of whatever the alignments counted.
                let observed = counter
                    .word_pron_counts
                    .get(&wid)
                    .and_then(|m| m.get(&pi))
                    .copied()
                    .unwrap_or(0.0);
                e.insert(pi, observed + 1.0);
                all_prons.push((wid, pi));
            }
        }
    }

    // base.py:425-433: probability = count / max count over the word's pronunciations.
    let mut out: HashMap<String, Vec<PronProb>> = HashMap::new();
    let mut prob_of: HashMap<(u32, u32), f32> = HashMap::new();
    for (&wid, prons) in &lex_counts {
        let max_value = prons.values().cloned().fold(0.0f32, f32::max).max(1.0);
        for (&pi, &c) in prons {
            prob_of.insert((wid, pi), clamp_probability(c / max_value));
        }
    }

    // base.py:437-450: silence-after probability, smoothed towards the global rate.
    let mut silence_probabilities: HashMap<Token, f32> = HashMap::new();
    for &(wid, pi) in &all_prons {
        let tok = Token::Word {
            word: wid,
            pron: pi,
        };
        let count = counter.silence_following.get(&tok).copied().unwrap_or(0.0);
        let total_count = count
            + counter
                .non_silence_following
                .get(&tok)
                .copied()
                .unwrap_or(0.0);
        let w_p_silence_count = count + silence_probability * LAMBDA_2;
        silence_probabilities.insert(
            tok,
            clamp_probability(w_p_silence_count / (total_count + LAMBDA_2)),
        );
    }

    // base.py:452-462: expected silence/non-silence counts before each token, under the
    // silence-after probability of whatever preceded it.
    let mut bar_silence: HashMap<Token, f32> = HashMap::new();
    let mut bar_non_silence: HashMap<Token, f32> = HashMap::new();
    for (&(w_p1, w_p2), &(sil_c, nonsil_c)) in &counter.ngram {
        // base.py:457-460: unseen predecessors fall back to 0.01.
        let silence_prob = silence_probabilities.get(&w_p1).copied().unwrap_or(0.01);
        let total_count = sil_c + nonsil_c;
        *bar_silence.entry(w_p2).or_insert(0.0) += total_count * silence_prob;
        *bar_non_silence.entry(w_p2).or_insert(0.0) += total_count * (1.0 - silence_prob);
    }

    // base.py:463-482: the before-corrections, observed over expected.
    for &(wid, pi) in &all_prons {
        let tok = Token::Word {
            word: wid,
            pron: pi,
        };
        let sil_b = counter.silence_before.get(&tok).copied().unwrap_or(0.0);
        let nonsil_b = counter.non_silence_before.get(&tok).copied().unwrap_or(0.0);
        let sil_corr = format_correction(
            (sil_b + LAMBDA_3) / (bar_silence.get(&tok).copied().unwrap_or(0.0) + LAMBDA_3),
        );
        let nonsil_corr = format_correction(
            (nonsil_b + LAMBDA_3) / (bar_non_silence.get(&tok).copied().unwrap_or(0.0) + LAMBDA_3),
        );
        let v = out.entry(word_strs[wid as usize].to_string()).or_default();
        if v.len() <= pi as usize {
            v.resize(
                pi as usize + 1,
                PronProb {
                    prob: 1.0,
                    silence_after_prob: silence_probability,
                    silence_before_correction: 1.0,
                    non_silence_before_correction: 1.0,
                },
            );
        }
        v[pi as usize] = PronProb {
            prob: prob_of.get(&(wid, pi)).copied().unwrap_or(1.0),
            silence_after_prob: silence_probabilities
                .get(&tok)
                .copied()
                .unwrap_or(silence_probability),
            silence_before_correction: sil_corr,
            non_silence_before_correction: nonsil_corr,
        };
    }

    // base.py:504-521: the initial silence probability and the final corrections, from the
    // `<s>` / `</s>` sentinels.
    let initial_silence_count = counter
        .silence_before
        .get(&Token::Start)
        .copied()
        .unwrap_or(0.0)
        + silence_probability * LAMBDA_2;
    let initial_non_silence_count = counter
        .non_silence_before
        .get(&Token::Start)
        .copied()
        .unwrap_or(0.0)
        + (1.0 - silence_probability) * LAMBDA_2;
    let initial_silence_probability = clamp_probability(
        initial_silence_count / (initial_silence_count + initial_non_silence_count),
    );

    let final_silence_correction = format_correction(
        (counter
            .silence_before
            .get(&Token::End)
            .copied()
            .unwrap_or(0.0)
            + LAMBDA_3)
            / (bar_silence.get(&Token::End).copied().unwrap_or(0.0) + LAMBDA_3),
    );
    let final_non_silence_correction = format_correction(
        (counter
            .non_silence_before
            .get(&Token::End)
            .copied()
            .unwrap_or(0.0)
            + LAMBDA_3)
            / (bar_non_silence.get(&Token::End).copied().unwrap_or(0.0) + LAMBDA_3),
    );

    LexiconProbs {
        words: out,
        silence_prob: silence_probability,
        initial_silence_prob: initial_silence_probability,
        final_silence_correction,
        final_non_silence_correction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counter_of(seqs: &[(&[Token], &[bool])]) -> Counter {
        let mut c = Counter::default();
        for (s, b) in seqs {
            c.add_utterance(s, b);
        }
        c
    }

    fn w(word: u32, pron: u32) -> Token {
        Token::Word { word, pron }
    }

    #[test]
    fn counts_match_mfa_walk() {
        // "<s> a b </s>" with silence between a and b only.
        let seq = [Token::Start, w(0, 0), w(1, 0), Token::End];
        let sil = [false, true, false];
        let c = counter_of(&[(&seq[..], &sil[..])]);
        assert_eq!(c.non_silence_before[&w(0, 0)], 1.0);
        assert_eq!(c.silence_before[&w(1, 0)], 1.0);
        assert_eq!(c.silence_following[&w(0, 0)], 1.0);
        assert_eq!(c.non_silence_following[&w(1, 0)], 1.0);
        // <s> is never counted as a pronunciation, but it does follow-count.
        assert!(!c.word_pron_counts.contains_key(&u32::MAX));
        assert_eq!(c.word_pron_counts[&0][&0], 1.0);
        // Bigram (a,b) is a silence bigram; (b,</s>) a non-silence one.
        assert_eq!(c.ngram[&(w(0, 0), w(1, 0))], (1.0, 0.0));
        assert_eq!(c.ngram[&(w(1, 0), Token::End)], (0.0, 1.0));
    }

    #[test]
    fn formatting_matches_mfa() {
        assert_eq!(clamp_probability(0.0), 0.01);
        assert_eq!(clamp_probability(1.0), 0.99);
        assert_eq!(clamp_probability(0.4449), 0.44);
        assert_eq!(format_correction(-1.0), 0.01);
        assert_eq!(format_correction(1.234), 1.23);
    }

    /// End-to-end over a two-utterance toy corpus, checking the published globals
    /// against the same formulas computed by hand (`base.py:435`, `:504-521`).
    fn toy_corpus() -> (Corpus, Vec<Option<IntervalAlignment>>) {
        use viter_kaldi::types::{PhoneInterval, SymbolTable, Utterance, WordInterval};
        // phones: 0=<eps>, 1=sil, 2=a, 3=b
        let mk = |id: &str, words: &[&str]| Utterance {
            id: id.into(),
            speaker: "s".into(),
            audio: std::path::PathBuf::from("x.wav"),
            words: words.iter().map(|w| w.to_string()).collect(),
            prons: words
                .iter()
                .map(|w| vec![Pronunciation::plain(vec![if *w == "a" { 2 } else { 3 }])])
                .collect(),
            text: words.join(" "),
        };
        let corpus = Corpus {
            utts: vec![mk("u1", &["a", "b"]), mk("u2", &["a", "b"])],
            speakers: vec!["s".into()],
            phones: SymbolTable::new(),
            silence_phones: vec![1],
            oov_words: Default::default(),
        };
        // u1: a  sil  b        u2: a  b
        let ia1 = IntervalAlignment {
            utt: "u1".into(),
            frame_shift_s: 0.01,
            phones: vec![
                PhoneInterval {
                    phone: 2,
                    start_frame: 0,
                    end_frame: 5,
                },
                PhoneInterval {
                    phone: 1,
                    start_frame: 5,
                    end_frame: 9,
                },
                PhoneInterval {
                    phone: 3,
                    start_frame: 9,
                    end_frame: 14,
                },
            ],
            words: vec![
                WordInterval {
                    word: 0,
                    pron: 0,
                    start_frame: 0,
                    end_frame: 5,
                },
                WordInterval {
                    word: 1,
                    pron: 0,
                    start_frame: 9,
                    end_frame: 14,
                },
            ],
        };
        let ia2 = IntervalAlignment {
            utt: "u2".into(),
            frame_shift_s: 0.01,
            phones: vec![
                PhoneInterval {
                    phone: 2,
                    start_frame: 0,
                    end_frame: 5,
                },
                PhoneInterval {
                    phone: 3,
                    start_frame: 5,
                    end_frame: 10,
                },
            ],
            words: vec![
                WordInterval {
                    word: 0,
                    pron: 0,
                    start_frame: 0,
                    end_frame: 5,
                },
                WordInterval {
                    word: 1,
                    pron: 0,
                    start_frame: 5,
                    end_frame: 10,
                },
            ],
        };
        (corpus, vec![Some(ia1), Some(ia2)])
    }

    #[test]
    fn estimate_globals_match_hand_computation() {
        let (corpus, ias) = toy_corpus();
        let probs = estimate_from_intervals(&corpus, 1, &[0, 1], &ias);
        // Boundaries: before a (x2, both non-sil), before b (sil, non-sil),
        // before </s> (non-sil x2)  ->  silence 1 of 6.
        assert_eq!(probs.silence_prob, clamp_probability(1.0 / 6.0));
        // "a" is followed by silence once out of two: (1 + p*2) / (2 + 2).
        let pa = &probs.words["a"][0];
        assert_eq!(
            pa.silence_after_prob,
            clamp_probability((1.0 + probs.silence_prob * 2.0) / 4.0)
        );
        // "b" never has silence after it, and is followed twice (by `</s>`).
        let pb = &probs.words["b"][0];
        assert_eq!(
            pb.silence_after_prob,
            clamp_probability(probs.silence_prob * 2.0 / 4.0)
        );
        // Only one pronunciation each, so probability is 1.0 -> clamped to 0.99.
        assert_eq!(pa.prob, 0.99);
        // `<s>` has no predecessor, so only the smoothing terms contribute
        // (`base.py:504-513`): p*lambda_2 against (1-p)*lambda_2.
        let isc = probs.silence_prob * 2.0;
        let insc = (1.0 - probs.silence_prob) * 2.0;
        assert_eq!(
            probs.initial_silence_prob,
            clamp_probability(isc / (isc + insc))
        );
        // b precedes </s> twice with no silence, so the non-silence correction exceeds 1.
        assert!(probs.final_non_silence_correction > 1.0);
        assert!(probs.final_silence_correction <= 1.0);
    }

    #[test]
    fn apply_overrides_only_learned_indices() {
        let mut probs = LexiconProbs::default();
        probs.words.insert(
            "a".into(),
            vec![PronProb {
                prob: 0.5,
                silence_after_prob: 0.2,
                silence_before_correction: 1.5,
                non_silence_before_correction: 0.8,
            }],
        );
        let prons = vec![Pronunciation::plain(vec![3]), Pronunciation::plain(vec![4])];
        let out = apply(&probs, "a", &prons);
        assert_eq!(out[0].prob, Some(0.5));
        assert_eq!(out[0].silence_after_prob, Some(0.2));
        assert_eq!(out[1].prob, None);
        let untouched = apply(&probs, "zzz", &prons);
        assert_eq!(untouched[0].prob, None);
    }
}
