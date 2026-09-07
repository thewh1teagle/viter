//! Print per-phone gaussian allocation of a .viter model: `cargo run --release -p viter-kaldi --example inspect model.viter`
use viter_kaldi::model::AcousticModel;
fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let m = AcousticModel::load(std::path::Path::new(&path)).expect("load");
    let sil = m.tm.silence_pdfs(&m.silence_phones);
    let total: usize = (0..m.am.num_pdfs())
        .map(|p| m.am.pdf(p as u32).num_gauss())
        .sum();
    let sil_g: usize = sil.iter().map(|&p| m.am.pdf(p).num_gauss()).sum();
    println!(
        "pdfs {}  gaussians {}  silence pdfs {} with {} gaussians ({:.1}%)",
        m.am.num_pdfs(),
        total,
        sil.len(),
        sil_g,
        100.0 * sil_g as f64 / total as f64
    );
    for &p in &sil {
        println!("  sil pdf {p}: {} gauss", m.am.pdf(p).num_gauss());
    }
    // transition probs of silence self-loops
    let mut shown = 0;
    for t in 1..=m.tm.num_transition_ids() as u32 {
        if m.silence_phones.contains(&m.tm.transition_id_to_phone(t)) && shown < 12 {
            println!(
                "  tid {t} phone {} state {} self_loop {} logp {:.3}",
                m.tm.transition_id_to_phone(t),
                m.tm.transition_id_to_hmm_state(t),
                m.tm.is_self_loop(t),
                m.tm.get_transition_log_prob(t)
            );
            shown += 1;
        }
    }
}
