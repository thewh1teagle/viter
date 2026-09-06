//! `viter serve` — browser viewer for TextGrids + audio.

use std::path::PathBuf;

use anyhow::Context;
use clap::Args;
use viter_serve::server::{ServeOptions, serve};

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Directory of audio files and TextGrids, scanned recursively
    #[arg(value_name = "DIR", default_value = ".")]
    pub dir: PathBuf,

    /// Port to bind on 127.0.0.1 (0 picks a free one)
    #[arg(short, long, default_value_t = 7878)]
    pub port: u16,

    /// Do not open a browser window
    #[arg(long)]
    pub no_open: bool,
}

pub fn run(args: ServeArgs) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new().context("failed to start the tokio runtime")?;
    rt.block_on(serve(ServeOptions {
        dir: args.dir,
        port: args.port,
        open: !args.no_open,
    }))
}
