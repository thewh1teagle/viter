//! `viter` — train an acoustic model, force-align a corpus, view the result.

mod cli;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    cli::init_logging()?;
    let cli = cli::Cli::parse();
    cli.run()
}
