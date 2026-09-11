//! `serve`: the browser viewer, interruptible from Python.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;
use viter_serve::server::{ServeOptions, serve as serve_impl};

use crate::convert::err;

/// How often the calling thread asks Python whether a signal arrived. Short enough that
/// Ctrl-C feels immediate, long enough to cost nothing.
const POLL: Duration = Duration::from_millis(100);

/// Serve a browser viewer for the TextGrids and audio under `dir`.
///
/// Blocks until interrupted. Ctrl-C raises `KeyboardInterrupt` in the calling thread and
/// shuts the server down; the server itself runs on a tokio runtime in a background thread,
/// so it never touches the GIL.
#[pyfunction]
#[pyo3(signature = (dir, *, port = 7878, open = false, audio = None))]
pub fn serve(
    py: Python<'_>,
    dir: PathBuf,
    port: u16,
    open: bool,
    audio: Option<PathBuf>,
) -> PyResult<()> {
    let opts = ServeOptions {
        dir,
        host: std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
        port,
        open,
        audio,
    };

    // The runtime lives entirely on the worker thread: created there, dropped there. Tearing
    // it down from another thread while `block_on` is still inside it panics that thread.
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let done = Arc::new(AtomicBool::new(false));
    let result: Arc<std::sync::Mutex<Option<anyhow::Result<()>>>> =
        Arc::new(std::sync::Mutex::new(None));

    let worker = {
        let done = Arc::clone(&done);
        let result = Arc::clone(&result);
        std::thread::spawn(move || {
            let r = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt.block_on(async {
                    // Whichever finishes first wins: the server returning on its own, or the
                    // stop signal. Dropping the server future cancels the listener and every
                    // in-flight request.
                    tokio::select! {
                        r = serve_impl(opts) => r,
                        _ = stop_rx => Ok(()),
                    }
                }),
                Err(e) => Err(anyhow::anyhow!("failed to start the tokio runtime: {e}")),
            };
            *result.lock().unwrap() = Some(r);
            done.store(true, Ordering::SeqCst);
        })
    };

    let mut interrupt = None;
    while !done.load(Ordering::SeqCst) {
        // Signals can only be checked while attached, so the sleep happens detached and the
        // check happens between sleeps.
        py.detach(|| std::thread::sleep(POLL));
        if let Err(e) = py.check_signals() {
            let _ = stop_tx.send(());
            interrupt = Some(e);
            break;
        }
    }
    // Let the worker finish its shutdown (and drop its runtime) before returning, so no
    // request outlives this call.
    py.detach(move || {
        let _ = worker.join();
    });

    if let Some(e) = interrupt {
        return Err(e);
    }
    match result.lock().unwrap().take() {
        Some(Ok(())) | None => Ok(()),
        Some(Err(e)) => Err(err(e)),
    }
}
