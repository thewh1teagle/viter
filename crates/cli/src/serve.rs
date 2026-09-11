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

    /// Port to bind (0 picks a free one)
    #[arg(short, long, default_value_t = 7878)]
    pub port: u16,

    /// Address to bind; 0.0.0.0 exposes the viewer to the network
    #[arg(long, default_value = "127.0.0.1", value_name = "ADDR")]
    pub host: std::net::IpAddr,

    /// Open the viewer in the default browser once the server is listening
    #[arg(long)]
    pub open: bool,

    /// Folder with the audio files when DIR holds only TextGrids (e.g. a training
    /// output next to its corpus); matched by file name
    #[arg(long, value_name = "DIR")]
    pub audio: Option<PathBuf>,
}

pub fn run(args: ServeArgs) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new().context("failed to start the tokio runtime")?;
    rt.block_on(serve(ServeOptions {
        dir: args.dir,
        host: args.host,
        port: args.port,
        open: args.open,
        audio: args.audio,
    }))
}
