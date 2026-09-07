//! `viter` — train an acoustic model, force-align a corpus, view the result.

fn main() -> anyhow::Result<()> {
    viter_cli::init_logging()?;
    viter_cli::run(std::env::args_os())
}
