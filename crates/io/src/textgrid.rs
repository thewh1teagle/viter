//! Praat TextGrid reading and writing.
//!
//! The long-format writer matches praatio's `_tgToLongTextForm`
//! (`plans/praatio/praatio/utilities/textgrid_io.py`) byte for byte, including its
//! number formatting (`my_math.numToStr`) and trailing-space quirks. Both long and
//! short text formats are read, in UTF-8 or UTF-16 (BOM sniffed via `encoding_rs`).

use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;
use viter_kaldi::types::{IntervalAlignment, SymbolTable, untag_phone};

/// One labelled interval.
#[derive(Clone, Debug, PartialEq)]
pub struct Interval {
    pub xmin: f64,
    pub xmax: f64,
    pub text: String,
}

/// An interval tier ("IntervalTier" in Praat).
#[derive(Clone, Debug, PartialEq)]
pub struct IntervalTier {
    pub name: String,
    pub xmin: f64,
    pub xmax: f64,
    pub intervals: Vec<Interval>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextGrid {
    pub xmin: f64,
    pub xmax: f64,
    pub tiers: Vec<IntervalTier>,
}

impl TextGrid {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading TextGrid {}", path.display()))?;
        let text = decode_bytes(&bytes);
        Self::from_str(&text).with_context(|| format!("parsing TextGrid {}", path.display()))
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        std::fs::write(path, self.to_string_long())
            .with_context(|| format!("writing TextGrid {}", path.display()))
    }

    /// Serialize in Praat's long text format, exactly as praatio writes it.
    pub fn to_string_long(&self) -> String {
        let tab = "    ";
        let mut out = String::new();
        out.push_str("File type = \"ooTextFile\"\n");
        out.push_str("Object class = \"TextGrid\"\n\n");
        out.push_str(&format!("xmin = {} \n", num_to_str(self.xmin)));
        out.push_str(&format!("xmax = {} \n", num_to_str(self.xmax)));
        out.push_str("tiers? <exists> \n");
        out.push_str(&format!("size = {} \n", self.tiers.len()));
        out.push_str("item []: \n");
        for (i, tier) in self.tiers.iter().enumerate() {
            out.push_str(&format!("{tab}item [{}]:\n", i + 1));
            out.push_str(&format!("{tab}{tab}class = \"IntervalTier\" \n"));
            out.push_str(&format!(
                "{tab}{tab}name = \"{}\" \n",
                escape_quotes(&tier.name)
            ));
            out.push_str(&format!("{tab}{tab}xmin = {} \n", num_to_str(tier.xmin)));
            out.push_str(&format!("{tab}{tab}xmax = {} \n", num_to_str(tier.xmax)));
            out.push_str(&format!(
                "{tab}{tab}intervals: size = {} \n",
                tier.intervals.len()
            ));
            for (j, iv) in tier.intervals.iter().enumerate() {
                out.push_str(&format!("{tab}{tab}intervals [{}]:\n", j + 1));
                out.push_str(&format!("{tab}{tab}{tab}xmin = {} \n", num_to_str(iv.xmin)));
                out.push_str(&format!("{tab}{tab}{tab}xmax = {} \n", num_to_str(iv.xmax)));
                out.push_str(&format!(
                    "{tab}{tab}{tab}text = \"{}\" \n",
                    escape_quotes(&iv.text)
                ));
            }
        }
        out
    }

    /// Serialize in Praat's short text format (praatio `_tgToShortTextForm`).
    pub fn to_string_short(&self) -> String {
        let mut out = String::new();
        out.push_str("File type = \"ooTextFile\"\n");
        out.push_str("Object class = \"TextGrid\"\n\n");
        out.push_str(&format!(
            "{}\n{}\n",
            num_to_str(self.xmin),
            num_to_str(self.xmax)
        ));
        out.push_str(&format!("<exists>\n{}\n", self.tiers.len()));
        for tier in &self.tiers {
            out.push_str("\"IntervalTier\"\n");
            out.push_str(&format!("\"{}\"\n", escape_quotes(&tier.name)));
            out.push_str(&format!(
                "{}\n{}\n{}\n",
                num_to_str(tier.xmin),
                num_to_str(tier.xmax),
                tier.intervals.len()
            ));
            for iv in &tier.intervals {
                out.push_str(&format!(
                    "{}\n{}\n\"{}\"\n",
                    num_to_str(iv.xmin),
                    num_to_str(iv.xmax),
                    escape_quotes(&iv.text)
                ));
            }
        }
        out
    }

    /// Parse either the long or the short text format.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        let data = s.replace("\r\n", "\n");
        let data = data.trim_start_matches('\u{feff}');
        if data.contains("ooTextFile short") || !data.contains("item [") {
            parse_short(data)
        } else {
            parse_long(data)
        }
    }

    pub fn tier(&self, name: &str) -> Option<&IntervalTier> {
        self.tiers.iter().find(|t| t.name == name)
    }
}

/// praatio's `my_math.numToStr`: integral values print as integers, everything else as
/// Python's `repr(float)` (shortest representation that round-trips), which Rust's
/// `{}` for `f64` also produces.
fn num_to_str(x: f64) -> String {
    let r = x.round();
    if is_close(x, r) {
        // "%d" % x truncates toward zero in Python; for values isclose to an integer,
        // truncation and rounding agree except for the -0.0 sign, which Praat writes as 0.
        let i = x as i64;
        if i == 0 {
            "0".to_string()
        } else {
            i.to_string()
        }
    } else {
        format_repr(x)
    }
}

/// Python `math.isclose(a, b, rel_tol=1e-14)`, as praatio uses.
fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-14 * a.abs().max(b.abs())
}

/// Python's `repr(float)`. Rust's `{}` is already shortest-round-trip, but it prints
/// very large/small magnitudes in full decimal where Python uses exponent notation.
fn format_repr(x: f64) -> String {
    let a = x.abs();
    if a != 0.0 && (a >= 1e16 || a < 1e-4) {
        // Python switches to scientific notation outside [1e-4, 1e16).
        let s = format!("{x:e}");
        // Rust: "1.5e-7"; Python: "1.5e-07".
        match s.split_once('e') {
            Some((mant, exp)) => {
                let (sign, digits) = match exp.strip_prefix('-') {
                    Some(d) => ("-", d),
                    None => ("+", exp),
                };
                let mant = if mant.contains('.') {
                    mant.to_string()
                } else {
                    format!("{mant}.0")
                };
                format!("{mant}e{sign}{:0>2}", digits)
            }
            None => s,
        }
    } else {
        format!("{x}")
    }
}

fn escape_quotes(s: &str) -> String {
    s.replace('"', "\"\"")
}

/// Decode bytes as UTF-16 (LE/BE, by BOM) or UTF-8, stripping any BOM.
pub(crate) fn decode_bytes(bytes: &[u8]) -> String {
    let enc = if bytes.starts_with(&[0xFF, 0xFE]) {
        encoding_rs::UTF_16LE
    } else if bytes.starts_with(&[0xFE, 0xFF]) {
        encoding_rs::UTF_16BE
    } else {
        encoding_rs::UTF_8
    };
    let (cow, _, _) = enc.decode(bytes);
    cow.into_owned()
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Value after the first `=` on a line whose key matches, searching from `from`.
fn field_after<'a>(lines: &'a [&'a str], from: usize, key: &str) -> Option<(usize, &'a str)> {
    for (i, line) in lines.iter().enumerate().skip(from) {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix(key)
            && rest.trim_start().starts_with('=')
        {
            let v = rest.trim_start().trim_start_matches('=').trim();
            return Some((i, v));
        }
    }
    None
}

fn parse_num(v: &str) -> Result<f64> {
    v.trim()
        .trim_matches('"')
        .parse::<f64>()
        .map_err(|_| anyhow!("expected a number, got {v:?}"))
}

fn unquote(v: &str) -> String {
    let t = v.trim();
    let inner = t
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(t);
    inner.replace("\"\"", "\"").trim().to_string()
}

fn parse_long(data: &str) -> Result<TextGrid> {
    let lines: Vec<&str> = data.lines().collect();
    let (_, xmin) = field_after(&lines, 0, "xmin").ok_or_else(|| anyhow!("missing xmin"))?;
    let xmin = parse_num(xmin)?;
    let (xmax_i, xmax) = field_after(&lines, 0, "xmax").ok_or_else(|| anyhow!("missing xmax"))?;
    let xmax = parse_num(xmax)?;

    // Tier blocks start at each `item [N]:` line after the `item []:` header.
    let mut starts: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate().skip(xmax_i) {
        let t = line.trim();
        if t.starts_with("item [") && !t.starts_with("item []") {
            starts.push(i);
        }
    }
    let mut tiers = Vec::new();
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(lines.len());
        let block = &lines[start..end];
        if let Some(tier) = parse_long_tier(block)? {
            tiers.push(tier);
        }
    }
    Ok(TextGrid { xmin, xmax, tiers })
}

fn parse_long_tier(block: &[&str]) -> Result<Option<IntervalTier>> {
    let class = field_after(block, 0, "class").map(|(_, v)| unquote(v));
    // Point tiers ("TextTier") carry no intervals; skip them.
    if class.as_deref() != Some("IntervalTier") {
        return Ok(None);
    }
    let name = field_after(block, 0, "name")
        .map(|(_, v)| unquote(v))
        .unwrap_or_default();
    let (xmin_i, xmin) =
        field_after(block, 0, "xmin").ok_or_else(|| anyhow!("tier missing xmin"))?;
    let xmin = parse_num(xmin)?;
    let (xmax_i, xmax) =
        field_after(block, xmin_i, "xmax").ok_or_else(|| anyhow!("tier missing xmax"))?;
    let xmax = parse_num(xmax)?;

    let mut intervals = Vec::new();
    let mut i = xmax_i + 1;
    while i < block.len() {
        if !block[i].trim_start().starts_with("intervals [") {
            i += 1;
            continue;
        }
        let (a, s) =
            field_after(block, i + 1, "xmin").ok_or_else(|| anyhow!("interval missing xmin"))?;
        let (b, e) =
            field_after(block, a + 1, "xmax").ok_or_else(|| anyhow!("interval missing xmax"))?;
        let (c, txt) = read_text_field(block, b + 1)?;
        intervals.push(Interval {
            xmin: parse_num(s)?,
            xmax: parse_num(e)?,
            text: txt,
        });
        i = c + 1;
    }
    Ok(Some(IntervalTier {
        name,
        xmin,
        xmax,
        intervals,
    }))
}

/// Read a `text = "..."` field, which may span several lines (labels with newlines).
/// Returns the index of the last line consumed and the unescaped label.
fn read_text_field(block: &[&str], from: usize) -> Result<(usize, String)> {
    let (i, first) = field_after(block, from, "text").ok_or_else(|| anyhow!("missing text"))?;
    let raw = first.trim_start();
    if !raw.starts_with('"') {
        return Ok((i, unquote(raw)));
    }
    let mut buf = raw[1..].to_string();
    let mut idx = i;
    loop {
        // The label ends at a quote that is not part of a doubled ("") escape.
        if let Some(end) = find_closing_quote(&buf) {
            let inner = buf[..end].replace("\"\"", "\"");
            return Ok((idx, inner.trim().to_string()));
        }
        idx += 1;
        if idx >= block.len() {
            bail!("unterminated text field in TextGrid");
        }
        buf.push('\n');
        buf.push_str(block[idx]);
    }
}

/// Index of the first unescaped `"` in `s`.
fn find_closing_quote(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'"' {
            if i + 1 < b.len() && b[i + 1] == b'"' {
                i += 2;
                continue;
            }
            return Some(i);
        }
        i += 1;
    }
    None
}

fn parse_short(data: &str) -> Result<TextGrid> {
    // Rows: header (2 lines + blank), xmin, xmax, <exists>, size, then tiers.
    let mut rows: Vec<&str> = data
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    // Drop the two header lines.
    rows.retain(|l| !l.starts_with("File type") && !l.starts_with("Object class"));
    if rows.len() < 4 {
        bail!("short TextGrid is truncated");
    }
    let xmin = parse_num(rows[0])?;
    let xmax = parse_num(rows[1])?;
    let mut i = 2;
    if rows[i].contains("exists") {
        i += 1;
    }
    let n_tiers: usize = rows[i]
        .parse()
        .map_err(|_| anyhow!("expected tier count, got {:?}", rows[i]))?;
    i += 1;

    let mut tiers = Vec::new();
    for _ in 0..n_tiers {
        if i + 4 > rows.len() {
            break;
        }
        let class = unquote(rows[i]);
        let name = unquote(rows[i + 1]);
        let t_min = parse_num(rows[i + 2])?;
        let t_max = parse_num(rows[i + 3])?;
        let count: usize = rows[i + 4]
            .parse()
            .map_err(|_| anyhow!("expected entry count, got {:?}", rows[i + 4]))?;
        i += 5;
        let is_interval = class == "IntervalTier";
        let stride = if is_interval { 3 } else { 2 };
        let mut intervals = Vec::new();
        for k in 0..count {
            let base = i + k * stride;
            if base + stride > rows.len() {
                break;
            }
            if is_interval {
                intervals.push(Interval {
                    xmin: parse_num(rows[base])?,
                    xmax: parse_num(rows[base + 1])?,
                    text: unquote(rows[base + 2]),
                });
            }
        }
        i += count * stride;
        if is_interval {
            tiers.push(IntervalTier {
                name,
                xmin: t_min,
                xmax: t_max,
                intervals,
            });
        }
    }
    Ok(TextGrid { xmin, xmax, tiers })
}

// ---------------------------------------------------------------------------
// Alignment -> TextGrid
// ---------------------------------------------------------------------------

/// Build an MFA-style TextGrid with a "words" tier and a "phones" tier.
///
/// Phones are untagged; silence/oov phones stay as `sil`/`spn` on the phones tier and
/// appear as empty intervals on the words tier (MFA `export_textgrid`). Gaps are filled
/// with empty intervals so both tiers run contiguously from 0 to `duration_s`.
pub fn from_alignment(
    a: &IntervalAlignment,
    phones: &SymbolTable,
    words: &[String],
    duration_s: f64,
) -> TextGrid {
    // Frame shift as an exact decimal (10 ms), not the f32 approximation, so
    // boundaries print as 0.08 rather than 0.07999999821186066 like MFA/praatio.
    let shift = (a.frame_shift_s as f64 * 1e6).round() / 1e6;
    let duration = round6(duration_s.max(0.0));
    // MFA snaps a final interval that lands within two (10 ms) frames of the end;
    // refined alignments count in 1 ms ticks, so the tolerance is fixed, not `shift`.
    let snap = 0.02;

    let mut phone_ivs: Vec<Interval> = a
        .phones
        .iter()
        .map(|p| Interval {
            xmin: round6(p.start_frame as f64 * shift),
            xmax: round6(p.end_frame as f64 * shift),
            text: untag_phone(phones.sym(p.phone)).to_string(),
        })
        .collect();

    let mut word_ivs: Vec<Interval> = a
        .words
        .iter()
        .map(|w| Interval {
            xmin: round6(w.start_frame as f64 * shift),
            xmax: round6(w.end_frame as f64 * shift),
            text: words.get(w.word as usize).cloned().unwrap_or_default(),
        })
        .collect();

    for ivs in [&mut phone_ivs, &mut word_ivs] {
        // MFA snaps a final interval that lands within two frames of the end.
        if let Some(last) = ivs.last_mut()
            && duration - last.xmax < snap
        {
            last.xmax = duration;
        }
    }

    let phones_tier = make_tier("phones", phone_ivs, duration);
    let words_tier = make_tier("words", word_ivs, duration);

    TextGrid {
        xmin: 0.0,
        xmax: duration,
        tiers: vec![words_tier, phones_tier],
    }
}

/// Fill gaps with empty intervals and clamp to `[0, duration]` (praatio `_fillInBlanks`).
fn make_tier(name: &str, ivs: Vec<Interval>, duration: f64) -> IntervalTier {
    let mut out: Vec<Interval> = Vec::with_capacity(ivs.len() + 2);
    let mut prev_end = 0.0f64;
    for mut iv in ivs {
        if iv.xmax > duration {
            iv.xmax = duration;
        }
        if iv.xmin < prev_end {
            iv.xmin = prev_end;
        }
        if iv.xmax <= iv.xmin {
            continue;
        }
        if iv.xmin > prev_end {
            out.push(Interval {
                xmin: prev_end,
                xmax: iv.xmin,
                text: String::new(),
            });
        }
        prev_end = iv.xmax;
        out.push(iv);
    }
    if prev_end < duration {
        out.push(Interval {
            xmin: prev_end,
            xmax: duration,
            text: String::new(),
        });
    }
    if out.is_empty() {
        out.push(Interval {
            xmin: 0.0,
            xmax: duration,
            text: String::new(),
        });
    }
    IntervalTier {
        name: name.to_string(),
        xmin: 0.0,
        xmax: duration,
        intervals: out,
    }
}

/// MFA rounds the file duration to 6 decimals before writing.
fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use viter_kaldi::types::{PhoneInterval, WordInterval};

    fn tg() -> TextGrid {
        TextGrid {
            xmin: 0.0,
            xmax: 1.5,
            tiers: vec![IntervalTier {
                name: "words".to_string(),
                xmin: 0.0,
                xmax: 1.5,
                intervals: vec![
                    Interval {
                        xmin: 0.0,
                        xmax: 0.5,
                        text: String::new(),
                    },
                    Interval {
                        xmin: 0.5,
                        xmax: 1.5,
                        text: "hello".to_string(),
                    },
                ],
            }],
        }
    }

    #[test]
    fn long_format_bytes() {
        let s = tg().to_string_long();
        let expected = concat!(
            "File type = \"ooTextFile\"\n",
            "Object class = \"TextGrid\"\n",
            "\n",
            "xmin = 0 \n",
            "xmax = 1.5 \n",
            "tiers? <exists> \n",
            "size = 1 \n",
            "item []: \n",
            "    item [1]:\n",
            "        class = \"IntervalTier\" \n",
            "        name = \"words\" \n",
            "        xmin = 0 \n",
            "        xmax = 1.5 \n",
            "        intervals: size = 2 \n",
            "        intervals [1]:\n",
            "            xmin = 0 \n",
            "            xmax = 0.5 \n",
            "            text = \"\" \n",
            "        intervals [2]:\n",
            "            xmin = 0.5 \n",
            "            xmax = 1.5 \n",
            "            text = \"hello\" \n",
        );
        assert_eq!(s, expected);
    }

    #[test]
    fn num_formatting() {
        assert_eq!(num_to_str(0.0), "0");
        assert_eq!(num_to_str(-0.0), "0");
        assert_eq!(num_to_str(3.0), "3");
        assert_eq!(num_to_str(1.5), "1.5");
        assert_eq!(num_to_str(0.3154201182247563), "0.3154201182247563");
    }

    #[test]
    fn long_roundtrip() {
        let original = tg();
        let parsed = TextGrid::from_str(&original.to_string_long()).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn short_roundtrip() {
        let original = tg();
        let parsed = TextGrid::from_str(&original.to_string_short()).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn quotes_are_escaped_and_unescaped() {
        let mut t = tg();
        t.tiers[0].intervals[1].text = "say \"hi\"".to_string();
        let parsed = TextGrid::from_str(&t.to_string_long()).unwrap();
        assert_eq!(parsed.tiers[0].intervals[1].text, "say \"hi\"");
    }

    #[test]
    fn utf16_is_decoded() {
        let s = tg().to_string_long();
        let mut bytes = vec![0xFF, 0xFE];
        for u in s.encode_utf16() {
            bytes.extend_from_slice(&u.to_le_bytes());
        }
        assert_eq!(decode_bytes(&bytes), s);
    }

    #[test]
    fn from_alignment_fills_gaps() {
        let a = IntervalAlignment {
            utt: "u".to_string(),
            frame_shift_s: 0.01,
            phones: vec![
                PhoneInterval {
                    phone: 1,
                    start_frame: 0,
                    end_frame: 10,
                },
                PhoneInterval {
                    phone: 3,
                    start_frame: 10,
                    end_frame: 30,
                },
            ],
            words: vec![WordInterval {
                word: 0,
                pron: 0,
                start_frame: 10,
                end_frame: 30,
            }],
        };
        let mut st = SymbolTable::new();
        st.add("sil");
        st.add("spn");
        st.add("AH_S");
        let tgrid = from_alignment(&a, &st, &["hello".to_string()], 0.3);

        assert_eq!(tgrid.tiers[0].name, "words");
        assert_eq!(tgrid.tiers[1].name, "phones");
        // words tier: leading empty interval then the word
        assert_eq!(tgrid.tiers[0].intervals.len(), 2);
        assert_eq!(tgrid.tiers[0].intervals[0].text, "");
        assert_eq!(tgrid.tiers[0].intervals[1].text, "hello");
        // phones tier: sil kept, tag stripped, contiguous to duration
        assert_eq!(tgrid.tiers[1].intervals[0].text, "sil");
        assert_eq!(tgrid.tiers[1].intervals[1].text, "AH");
        for tier in &tgrid.tiers {
            assert_eq!(tier.intervals.first().unwrap().xmin, 0.0);
            assert!((tier.intervals.last().unwrap().xmax - 0.3).abs() < 1e-9);
            for w in tier.intervals.windows(2) {
                assert_eq!(w[0].xmax, w[1].xmin);
            }
        }
    }
}
