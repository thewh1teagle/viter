//! JSON API handlers: file listing, TextGrid, audio streaming with Range, waveform peaks.
//!
//! Every route is keyed by an `id`, which is the audio/TextGrid path relative to the served
//! directory with its extension stripped (e.g. `speaker1/utt_003`). Because ids contain `/`,
//! all id routes use axum's `{*id}` wildcard capture.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Path as AxPath, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// Audio extensions we pair with a `.TextGrid`, in preference order.
const AUDIO_EXTS: [&str; 3] = ["wav", "flac", "mp3"];

/// One aligned pair of audio + TextGrid, as returned by `GET /api/files`.
#[derive(Clone, Debug, Serialize)]
pub struct FileEntry {
    /// Relative path without extension; the key for every other route.
    pub id: String,
    /// Audio file path relative to the served directory.
    pub audio: String,
    /// TextGrid path relative to the served directory, if one exists.
    pub textgrid: Option<String>,
    /// Duration in seconds, read from the TextGrid's `xmax` when available.
    pub duration: Option<f64>,
}

/// Absolute paths behind one [`FileEntry`].
#[derive(Clone, Debug)]
struct Entry {
    meta: FileEntry,
    audio_abs: PathBuf,
    textgrid_abs: Option<PathBuf>,
}

/// Shared server state: the served directory plus caches for the scan and the peak arrays.
#[derive(Clone)]
pub struct AppState {
    root: PathBuf,
    /// Optional folder to take audio from when a TextGrid has no sibling audio
    /// (a training output directory next to its corpus). Matched by file stem.
    audio_root: Option<PathBuf>,
    /// Cached directory scan, keyed by id. Rebuilt on demand when stale.
    index: Arc<Mutex<Index>>,
    /// Peaks cache keyed by `(id, px)`.
    peaks: Arc<Mutex<HashMap<(String, u32), Arc<PeaksResponse>>>>,
}

struct Index {
    entries: HashMap<String, Entry>,
    /// Ids in stable sorted order, for a deterministic listing.
    order: Vec<String>,
    /// When the scan was taken; a re-scan is cheap enough to redo every few seconds.
    scanned_at: std::time::Instant,
}

/// Minimum age before `GET /api/files` re-scans the directory.
const RESCAN_AFTER: std::time::Duration = std::time::Duration::from_secs(2);

impl AppState {
    /// Scan `root` once and build the initial index.
    pub fn new(root: PathBuf, audio_root: Option<PathBuf>) -> anyhow::Result<Self> {
        let (entries, order) = scan_dir(&root, audio_root.as_deref())?;
        Ok(Self {
            root,
            audio_root,
            index: Arc::new(Mutex::new(Index {
                entries,
                order,
                scanned_at: std::time::Instant::now(),
            })),
            peaks: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Number of paired files found by the most recent scan.
    pub fn num_files(&self) -> usize {
        self.index.lock().expect("index poisoned").order.len()
    }

    /// Re-scan if the cached index is older than [`RESCAN_AFTER`]. Scan failures keep the
    /// previous index rather than taking down the request.
    fn refresh_if_stale(&self) {
        let mut idx = self.index.lock().expect("index poisoned");
        if idx.scanned_at.elapsed() < RESCAN_AFTER {
            return;
        }
        match scan_dir(&self.root, self.audio_root.as_deref()) {
            Ok((entries, order)) => {
                idx.entries = entries;
                idx.order = order;
            }
            Err(e) => tracing::warn!(error = %e, "re-scan of {} failed", self.root.display()),
        }
        idx.scanned_at = std::time::Instant::now();
    }

    /// Look up one entry by id, re-scanning once if it is missing (a file may have just been
    /// written by a concurrent `viter align`).
    fn entry(&self, id: &str) -> Option<Entry> {
        {
            let idx = self.index.lock().expect("index poisoned");
            if let Some(e) = idx.entries.get(id) {
                return Some(e.clone());
            }
        }
        self.refresh_if_stale();
        let idx = self.index.lock().expect("index poisoned");
        idx.entries.get(id).cloned()
    }
}

/// Walk `root` recursively, pairing each `x.TextGrid` and each audio file by their stem.
///
/// An entry is produced for every audio file found; the TextGrid is attached when a sibling of
/// the same stem exists, so freshly recorded audio still shows up in the viewer before it has
/// been aligned.
fn scan_dir(
    root: &Path,
    audio_root: Option<&Path>,
) -> anyhow::Result<(HashMap<String, Entry>, Vec<String>)> {
    let mut audio: HashMap<String, PathBuf> = HashMap::new();
    let mut grids: HashMap<String, PathBuf> = HashMap::new();
    walk(root, root, &mut audio, &mut grids)?;

    // TextGrids without sibling audio: look the audio up by stem in `audio_root`.
    if let Some(aroot) = audio_root {
        let mut ext_audio: HashMap<String, PathBuf> = HashMap::new();
        let mut ext_grids: HashMap<String, PathBuf> = HashMap::new();
        walk(aroot, aroot, &mut ext_audio, &mut ext_grids)?;
        let by_stem: HashMap<String, PathBuf> = ext_audio
            .into_values()
            .filter_map(|p| Some((p.file_stem()?.to_string_lossy().into_owned(), p)))
            .collect();
        for (id, g) in &grids {
            if audio.contains_key(id) {
                continue;
            }
            let stem = g
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            if let Some(a) = by_stem.get(&stem) {
                audio.insert(id.clone(), a.clone());
            }
        }
    }

    let mut entries = HashMap::with_capacity(audio.len());
    for (id, audio_abs) in audio {
        let textgrid_abs = grids.get(&id).cloned();
        let duration = textgrid_abs.as_deref().and_then(textgrid_duration);
        let meta = FileEntry {
            audio: audio_abs
                .strip_prefix(root)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| audio_abs.to_string_lossy().into_owned()),
            textgrid: textgrid_abs.as_deref().map(|p| rel_string(root, p)),
            duration,
            id: id.clone(),
        };
        entries.insert(
            id,
            Entry {
                meta,
                audio_abs,
                textgrid_abs,
            },
        );
    }
    let mut order: Vec<String> = entries.keys().cloned().collect();
    order.sort();
    Ok((entries, order))
}

/// Recursive directory walk. Symlinked directories are not followed, so a self-referential
/// link inside the served folder cannot spin forever.
fn walk(
    root: &Path,
    dir: &Path,
    audio: &mut HashMap<String, PathBuf>,
    grids: &mut HashMap<String, PathBuf>,
) -> anyhow::Result<()> {
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            tracing::warn!(error = %e, "cannot read {}", dir.display());
            return Ok(());
        }
    };
    for ent in rd {
        let ent = match ent {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "cannot stat entry in {}", dir.display());
                continue;
            }
        };
        let path = ent.path();
        // Follow symlinks: corpora are commonly assembled from linked wavs.
        let ty = match std::fs::metadata(&path) {
            Ok(m) => m.file_type(),
            Err(_) => continue,
        };
        if ty.is_dir() {
            // Skip dotted directories (.git, .cache) — never corpus content.
            if path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with('.'))
            {
                continue;
            }
            walk(root, &path, audio, grids)?;
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        let Some(id) = id_for(root, &path) else {
            continue;
        };
        if ext.eq_ignore_ascii_case("textgrid") {
            grids.insert(id, path);
        } else if let Some(pref) = AUDIO_EXTS.iter().position(|a| ext.eq_ignore_ascii_case(a)) {
            // Prefer wav over flac over mp3 when several encodings of one utterance coexist.
            match audio.get(&id) {
                Some(prev) => {
                    let prev_pref = prev
                        .extension()
                        .and_then(|e| e.to_str())
                        .and_then(|e| AUDIO_EXTS.iter().position(|a| e.eq_ignore_ascii_case(a)))
                        .unwrap_or(usize::MAX);
                    if pref < prev_pref {
                        audio.insert(id, path);
                    }
                }
                None => {
                    audio.insert(id, path);
                }
            }
        }
    }
    Ok(())
}

/// Id for a file: its path relative to `root`, without extension, with `/` separators.
fn id_for(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    let parent = rel.parent().filter(|p| !p.as_os_str().is_empty());
    let stem = rel.file_stem()?.to_str()?;
    Some(match parent {
        Some(p) => format!("{}/{}", p.to_string_lossy().replace('\\', "/"), stem),
        None => stem.to_string(),
    })
}

fn rel_string(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Cheap duration probe: a TextGrid's `xmax` is the audio duration MFA wrote at export time,
/// so the listing needs no audio decoding.
fn textgrid_duration(path: &Path) -> Option<f64> {
    let tg = viter_io::textgrid::TextGrid::read(path).ok()?;
    Some(tg.xmax)
}

// ---------------------------------------------------------------------------
// GET /api/files
// ---------------------------------------------------------------------------

/// List every audio file under the served directory, with its TextGrid when aligned.
pub async fn list_files(State(st): State<AppState>) -> Response {
    st.refresh_if_stale();
    let idx = st.index.lock().expect("index poisoned");
    let files: Vec<&FileEntry> = idx
        .order
        .iter()
        .filter_map(|id| idx.entries.get(id))
        .map(|e| &e.meta)
        .collect();
    json_ok(&files)
}

// ---------------------------------------------------------------------------
// GET /api/textgrid/{*id}
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct TgResponse<'a> {
    xmin: f64,
    xmax: f64,
    tiers: Vec<TgTier<'a>>,
}

#[derive(Serialize)]
struct TgTier<'a> {
    name: &'a str,
    xmin: f64,
    xmax: f64,
    intervals: Vec<TgInterval<'a>>,
}

#[derive(Serialize)]
struct TgInterval<'a> {
    xmin: f64,
    xmax: f64,
    text: &'a str,
}

/// Parse and return one TextGrid as JSON.
pub async fn get_textgrid(State(st): State<AppState>, AxPath(id): AxPath<String>) -> Response {
    let Some(entry) = st.entry(&id) else {
        return err(StatusCode::NOT_FOUND, "no such file");
    };
    let Some(path) = entry.textgrid_abs else {
        return err(StatusCode::NOT_FOUND, "no TextGrid for this file");
    };
    let tg = match viter_io::textgrid::TextGrid::read(&path) {
        Ok(tg) => tg,
        Err(e) => {
            tracing::error!(error = %e, "failed to parse {}", path.display());
            return err(StatusCode::UNPROCESSABLE_ENTITY, "cannot parse TextGrid");
        }
    };
    let body = TgResponse {
        xmin: tg.xmin,
        xmax: tg.xmax,
        tiers: tg
            .tiers
            .iter()
            .map(|t| TgTier {
                name: &t.name,
                xmin: t.xmin,
                xmax: t.xmax,
                intervals: t
                    .intervals
                    .iter()
                    .map(|i| TgInterval {
                        xmin: i.xmin,
                        xmax: i.xmax,
                        text: &i.text,
                    })
                    .collect(),
            })
            .collect(),
    };
    json_ok(&body)
}

// ---------------------------------------------------------------------------
// GET /api/audio/{*id}  — byte serving with Range support
// ---------------------------------------------------------------------------

/// Stream the audio file, honouring a single-range `Range: bytes=` header with a 206 response.
///
/// Multi-range requests are answered with the whole file (a legal, if unhelpful, response);
/// browsers' media elements only ever ask for one range.
pub async fn get_audio(
    State(st): State<AppState>,
    AxPath(id): AxPath<String>,
    headers: HeaderMap,
) -> Response {
    let Some(entry) = st.entry(&id) else {
        return err(StatusCode::NOT_FOUND, "no such file");
    };
    let path = entry.audio_abs;
    let mime = mime_guess::from_path(&path)
        .first_or_octet_stream()
        .to_string();

    let len = match std::fs::metadata(&path) {
        Ok(m) => m.len(),
        Err(e) => {
            tracing::error!(error = %e, "cannot stat {}", path.display());
            return err(StatusCode::NOT_FOUND, "audio file unreadable");
        }
    };

    let range = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| parse_range(v, len));

    match range {
        Some(Err(())) => {
            // Syntactically valid but unsatisfiable: RFC 9110 wants 416 + Content-Range.
            let mut resp = err(StatusCode::RANGE_NOT_SATISFIABLE, "range not satisfiable");
            if let Ok(v) = HeaderValue::from_str(&format!("bytes */{len}")) {
                resp.headers_mut().insert(header::CONTENT_RANGE, v);
            }
            resp
        }
        Some(Ok((start, end))) => {
            let count = end - start + 1;
            let bytes = match read_span(&path, start, count) {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!(error = %e, "cannot read range of {}", path.display());
                    return err(StatusCode::INTERNAL_SERVER_ERROR, "read failed");
                }
            };
            let mut resp = Response::new(Body::from(bytes));
            *resp.status_mut() = StatusCode::PARTIAL_CONTENT;
            let h = resp.headers_mut();
            insert_str(h, header::CONTENT_TYPE, &mime);
            insert_str(
                h,
                header::CONTENT_RANGE,
                &format!("bytes {start}-{end}/{len}"),
            );
            insert_str(h, header::CONTENT_LENGTH, &count.to_string());
            insert_str(h, header::ACCEPT_RANGES, "bytes");
            resp
        }
        None => {
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!(error = %e, "cannot read {}", path.display());
                    return err(StatusCode::INTERNAL_SERVER_ERROR, "read failed");
                }
            };
            let mut resp = Response::new(Body::from(bytes));
            let h = resp.headers_mut();
            insert_str(h, header::CONTENT_TYPE, &mime);
            insert_str(h, header::CONTENT_LENGTH, &len.to_string());
            insert_str(h, header::ACCEPT_RANGES, "bytes");
            resp
        }
    }
}

/// Parse an HTTP `Range` header for a resource of `len` bytes.
///
/// Returns `None` when the header is not a byte range we handle (so the caller serves the whole
/// body), `Some(Err(()))` when it is a byte range that cannot be satisfied, and
/// `Some(Ok((start, end)))` with an inclusive, clamped span otherwise.
fn parse_range(raw: &str, len: u64) -> Option<Result<(u64, u64), ()>> {
    let spec = raw.trim().strip_prefix("bytes=")?.trim();
    // Only the first range of a set is honoured; a comma means multi-range.
    if spec.contains(',') {
        return None;
    }
    let (a, b) = spec.split_once('-')?;
    let (a, b) = (a.trim(), b.trim());
    if len == 0 {
        return Some(Err(()));
    }
    let (start, end) = if a.is_empty() {
        // `-N`: the final N bytes.
        let n: u64 = b.parse().ok()?;
        if n == 0 {
            return Some(Err(()));
        }
        (len.saturating_sub(n), len - 1)
    } else {
        let start: u64 = a.parse().ok()?;
        if start >= len {
            return Some(Err(()));
        }
        let end = if b.is_empty() {
            len - 1
        } else {
            b.parse::<u64>().ok()?.min(len - 1)
        };
        (start, end)
    };
    if start > end {
        Some(Err(()))
    } else {
        Some(Ok((start, end)))
    }
}

/// Read `count` bytes starting at `start` without loading the whole file.
fn read_span(path: &Path, start: u64, count: u64) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    f.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0u8; count as usize];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// GET /api/peaks/{*id}?px=N
// ---------------------------------------------------------------------------

/// Min/max pairs per pixel column, ready to draw as a waveform.
#[derive(Debug, Serialize)]
pub struct PeaksResponse {
    pub sample_rate: u32,
    pub duration: f64,
    /// Flat `[min0, max0, min1, max1, ...]`, two values per pixel column, each in `-1..1`.
    pub peaks: Vec<f32>,
}

#[derive(serde::Deserialize)]
pub struct PeaksQuery {
    /// Number of pixel columns to reduce to. Defaults to 2000, clamped to 1..=20000.
    pub px: Option<u32>,
}

/// Decode the audio and reduce it to `px` min/max pairs, cached in memory per `(id, px)`.
pub async fn get_peaks(
    State(st): State<AppState>,
    AxPath(id): AxPath<String>,
    Query(q): Query<PeaksQuery>,
) -> Response {
    let px = q.px.unwrap_or(2000).clamp(1, 20_000);

    if let Some(hit) = st
        .peaks
        .lock()
        .expect("peaks poisoned")
        .get(&(id.clone(), px))
    {
        return json_ok(hit.as_ref());
    }
    let Some(entry) = st.entry(&id) else {
        return err(StatusCode::NOT_FOUND, "no such file");
    };
    let path = entry.audio_abs;

    // Decoding is blocking and can take a moment for long files; keep it off the async runtime.
    let decoded = tokio::task::spawn_blocking(move || compute_peaks(&path, px)).await;
    let peaks = match decoded {
        Ok(Ok(p)) => Arc::new(p),
        Ok(Err(e)) => {
            tracing::error!(error = %e, id = %id, "cannot decode audio for peaks");
            return err(StatusCode::UNPROCESSABLE_ENTITY, "cannot decode audio");
        }
        Err(e) => {
            tracing::error!(error = %e, "peak worker panicked");
            return err(StatusCode::INTERNAL_SERVER_ERROR, "peak computation failed");
        }
    };
    st.peaks
        .lock()
        .expect("peaks poisoned")
        .insert((id, px), Arc::clone(&peaks));
    json_ok(peaks.as_ref())
}

/// Decode `path` and reduce its samples to `px` min/max pairs.
fn compute_peaks(path: &Path, px: u32) -> anyhow::Result<PeaksResponse> {
    let audio = viter_kaldi::audio::read(path)?;
    let n = audio.samples.len();
    let duration = if audio.sample_rate == 0 {
        0.0
    } else {
        n as f64 / audio.sample_rate as f64
    };
    let cols = px as usize;
    let mut peaks = Vec::with_capacity(cols * 2);

    if n == 0 {
        peaks.resize(cols * 2, 0.0);
        return Ok(PeaksResponse {
            sample_rate: audio.sample_rate,
            duration,
            peaks,
        });
    }
    for c in 0..cols {
        // Column boundaries by exact rational split, so no sample is dropped or double-counted.
        let start = c * n / cols;
        let end = ((c + 1) * n / cols).max(start + 1).min(n);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &s in &audio.samples[start..end] {
            if s < lo {
                lo = s;
            }
            if s > hi {
                hi = s;
            }
        }
        // A column past the end of a very short file gets a flat zero pair.
        if !lo.is_finite() || !hi.is_finite() {
            lo = 0.0;
            hi = 0.0;
        }
        peaks.push(lo);
        peaks.push(hi);
    }
    Ok(PeaksResponse {
        sample_rate: audio.sample_rate,
        duration,
        peaks,
    })
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn insert_str(h: &mut HeaderMap, key: header::HeaderName, value: &str) {
    if let Ok(v) = HeaderValue::from_str(value) {
        h.insert(key, v);
    }
}

/// Serialize to JSON, turning a serialization failure into a 500 rather than a panic.
fn json_ok<T: Serialize>(v: &T) -> Response {
    match serde_json::to_vec(v) {
        Ok(body) => {
            let mut resp = Response::new(Body::from(body));
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            resp
        }
        Err(e) => {
            tracing::error!(error = %e, "JSON serialization failed");
            err(StatusCode::INTERNAL_SERVER_ERROR, "serialization failed")
        }
    }
}

/// A JSON error body with the given status.
fn err(status: StatusCode, msg: &str) -> Response {
    let body = serde_json::json!({ "error": msg }).to_string();
    let mut resp = (status, body).into_response();
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_open_ended() {
        assert_eq!(parse_range("bytes=100-", 1000), Some(Ok((100, 999))));
    }

    #[test]
    fn range_closed_and_clamped() {
        assert_eq!(parse_range("bytes=0-99", 1000), Some(Ok((0, 99))));
        assert_eq!(parse_range("bytes=900-5000", 1000), Some(Ok((900, 999))));
    }

    #[test]
    fn range_suffix() {
        assert_eq!(parse_range("bytes=-100", 1000), Some(Ok((900, 999))));
        // A suffix longer than the file yields the whole file.
        assert_eq!(parse_range("bytes=-5000", 1000), Some(Ok((0, 999))));
    }

    #[test]
    fn range_unsatisfiable() {
        assert_eq!(parse_range("bytes=1000-", 1000), Some(Err(())));
        assert_eq!(parse_range("bytes=-0", 1000), Some(Err(())));
        assert_eq!(parse_range("bytes=0-0", 0), Some(Err(())));
    }

    #[test]
    fn range_ignored_forms() {
        assert_eq!(parse_range("items=0-10", 1000), None);
        assert_eq!(parse_range("bytes=0-10,20-30", 1000), None);
        assert_eq!(parse_range("bytes=abc", 1000), None);
    }

    #[test]
    fn ids_are_relative_paths_without_extension() {
        let root = Path::new("/corpus");
        assert_eq!(id_for(root, Path::new("/corpus/a.wav")).unwrap(), "a");
        assert_eq!(
            id_for(root, Path::new("/corpus/spk1/utt_003.TextGrid")).unwrap(),
            "spk1/utt_003"
        );
        assert_eq!(
            id_for(root, Path::new("/corpus/a/b/c.flac")).unwrap(),
            "a/b/c"
        );
    }
}
