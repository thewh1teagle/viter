//! The axum server: JSON API under `/api`, everything else the embedded React viewer.

use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use anyhow::Context;
use axum::Router;
use axum::body::Body;
use axum::extract::Path as AxPath;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::Response;
use axum::routing::get;
use rust_embed::RustEmbed;
use tower_http::cors::CorsLayer;

use crate::api::{self, AppState};

/// Options for [`serve`].
pub struct ServeOptions {
    /// Folder with wavs + TextGrids (typically an aligner output directory).
    pub dir: PathBuf,
    /// TCP port on loopback.
    pub port: u16,
    /// Open the URL in the default browser once the listener is bound.
    pub open: bool,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("."),
            port: 7878,
            open: true,
        }
    }
}

/// The built React viewer, compiled into the binary in release builds.
#[derive(RustEmbed)]
#[folder = "../../web/dist"]
struct Assets;

/// Serve `opts.dir` on `127.0.0.1:opts.port` until the process is interrupted.
pub async fn serve(opts: ServeOptions) -> anyhow::Result<()> {
    let dir = opts
        .dir
        .canonicalize()
        .with_context(|| format!("cannot open directory {}", opts.dir.display()))?;
    anyhow::ensure!(dir.is_dir(), "{} is not a directory", dir.display());

    let state =
        AppState::new(dir.clone()).with_context(|| format!("failed to scan {}", dir.display()))?;
    let num_files = state.num_files();

    let app = router(state);

    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, opts.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot bind {addr} (is another viter serve running?)"))?;
    // Port 0 means "any free port", so report what we actually got.
    let bound = listener.local_addr().context("cannot read local address")?;
    let url = format!("http://{bound}");

    print_banner(&url, &dir, num_files);

    if opts.open
        && let Err(e) = open::that(&url)
    {
        // Not fatal: headless machines and remote shells have no browser to open.
        tracing::warn!(error = %e, "could not open a browser; visit {url}");
    }

    axum::serve(listener, app).await.context("server error")?;
    Ok(())
}

/// Build the full router. Split out from [`serve`] so it can be exercised without binding a port.
pub fn router(state: AppState) -> Router {
    // Ids are relative paths and so contain `/`; the wildcard capture keeps them intact.
    Router::new()
        .route("/api/files", get(api::list_files))
        .route("/api/textgrid/{*id}", get(api::get_textgrid))
        .route("/api/audio/{*id}", get(api::get_audio))
        .route("/api/peaks/{*id}", get(api::get_peaks))
        .with_state(state)
        // The viewer is served from the same origin, but CORS keeps a separate `vite dev`
        // front-end (port 5173) able to talk to this API during development.
        .layer(CorsLayer::permissive())
        .route("/assets/{*path}", get(static_asset))
        .fallback(get(spa_fallback))
}

/// Serve one embedded asset by path, 404 when it is not in the bundle.
async fn static_asset(AxPath(path): AxPath<String>) -> Response {
    match embedded(&format!("assets/{path}")) {
        Some(resp) => resp,
        None => not_found(),
    }
}

/// Serve any other path from the bundle, falling back to `index.html` so client-side routes work.
async fn spa_fallback(uri: axum::http::Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };

    if let Some(resp) = embedded(path) {
        return resp;
    }
    // A missing file with an extension is a genuine 404; an extensionless path is a SPA route.
    if std::path::Path::new(path).extension().is_some() {
        return not_found();
    }
    match embedded("index.html") {
        Some(resp) => resp,
        None => {
            tracing::error!(
                "web/dist/index.html is missing from the binary; run `pnpm build` in web/"
            );
            let mut resp = Response::new(Body::from(
                "The viewer was not built into this binary. Run `pnpm build` in web/ and rebuild.",
            ));
            *resp.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            );
            resp
        }
    }
}

/// Look up an embedded file and wrap it in a response with its mime type.
fn embedded(path: &str) -> Option<Response> {
    let file = Assets::get(path)?;
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    let mut resp = Response::new(Body::from(file.data.into_owned()));
    let h = resp.headers_mut();
    if let Ok(v) = HeaderValue::from_str(mime.as_ref()) {
        h.insert(header::CONTENT_TYPE, v);
    }
    // Vite fingerprints filenames under /assets, so those are safe to cache forever;
    // index.html must always be revalidated or a rebuild would never be picked up.
    let cache = if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
    Some(resp)
}

fn not_found() -> Response {
    let mut resp = Response::new(Body::from("not found"));
    *resp.status_mut() = StatusCode::NOT_FOUND;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    resp
}

/// Print the serving banner. This is the binary's user interface, so it uses stdout directly
/// rather than `tracing` (the no-println rule covers library logic, not the CLI surface).
fn print_banner(url: &str, dir: &std::path::Path, num_files: usize) {
    const BOLD: &str = "\x1b[1m";
    const CYAN: &str = "\x1b[36m";
    const DIM: &str = "\x1b[2m";
    const RESET: &str = "\x1b[0m";

    println!();
    println!("  {BOLD}viter{RESET} {DIM}viewer{RESET}");
    println!("  {DIM}{}{RESET}  {} file(s)", dir.display(), num_files);
    println!("  {BOLD}{CYAN}{url}{RESET}");
    println!("  {DIM}Ctrl-C to stop{RESET}");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_is_embedded() {
        // Guards against the web bundle going missing from the build.
        assert!(
            Assets::get("index.html").is_some(),
            "web/dist/index.html must be built"
        );
    }

    #[test]
    fn default_port_is_7878() {
        assert_eq!(ServeOptions::default().port, 7878);
    }
}
