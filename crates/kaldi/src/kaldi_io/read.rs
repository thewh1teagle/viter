//! The low-level Kaldi binary reader: tokens, basic types, integer vectors,
//! and `Vector`/`Matrix` blobs.
//!
//! Kaldi's binary stream starts with `"\0B"` (`InitKaldiOutputStream`,
//! `plans/kaldi/src/base/io-funcs-inl.h:291`). After that:
//!
//! * `WriteToken` writes the token text followed by a single space.
//! * `WriteBasicType` writes one byte holding `sizeof(T)`, then the value
//!   little-endian (`io-funcs-inl.h`). Floats/doubles are written the same way,
//!   with the size byte distinguishing `float` from `double`.
//! * `WriteIntegerVector` writes the element size byte, an `int32` count (raw,
//!   *no* size byte of its own), then the packed elements.
//! * `Vector::Write` / `Matrix::Write` emit the token `FV`/`DV`/`FM`/`DM`, then
//!   the dimensions via `WriteBasicType`, then raw row-major data.

use std::io;

/// Errors from parsing a Kaldi binary file.
#[derive(Debug, thiserror::Error)]
pub enum KaldiReadError {
    #[error("i/o error at offset {offset}: {source}")]
    Io {
        offset: usize,
        #[source]
        source: io::Error,
    },

    #[error("unexpected end of file at offset {offset}: need {need} more bytes")]
    Eof { offset: usize, need: usize },

    #[error("not a Kaldi binary file: expected a \\0B header, found {found:02x?}")]
    NotBinary { found: [u8; 2] },

    #[error("at offset {offset}: expected token {expected:?}, found {found:?}")]
    BadToken {
        offset: usize,
        expected: String,
        found: String,
    },

    #[error("at offset {offset}: expected a {want}-byte {what}, but the size byte says {got}")]
    BadWidth {
        offset: usize,
        what: &'static str,
        want: usize,
        got: u8,
    },

    #[error("at offset {offset}: {0}", .msg)]
    Malformed { offset: usize, msg: String },
}

type Result<T> = std::result::Result<T, KaldiReadError>;

/// A cursor over an in-memory Kaldi binary file.
pub struct KaldiReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> KaldiReader<'a> {
    /// Wrap `buf` and consume the `"\0B"` binary header.
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        if buf.len() < 2 || buf[0] != 0 || buf[1] != b'B' {
            let mut found = [0u8; 2];
            found.copy_from_slice(&buf[..2.min(buf.len())]);
            return Err(KaldiReadError::NotBinary { found });
        }
        Ok(Self { buf, pos: 2 })
    }

    pub fn offset(&self) -> usize {
        self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(KaldiReadError::Eof {
                offset: self.pos,
                need: self.pos + n - self.buf.len(),
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// The next non-whitespace byte, without consuming it. Kaldi's `Peek`
    /// skips whitespace even in binary mode.
    pub fn peek(&self) -> Option<u8> {
        self.buf[self.pos..]
            .iter()
            .copied()
            .find(|b| !b.is_ascii_whitespace())
    }

    fn skip_ws(&mut self) {
        while self.pos < self.buf.len() && self.buf[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    /// Kaldi `ReadToken`: skip whitespace, then read up to the next whitespace.
    pub fn token(&mut self) -> Result<String> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.buf.len() && !self.buf[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(KaldiReadError::Eof {
                offset: start,
                need: 1,
            });
        }
        let tok = String::from_utf8_lossy(&self.buf[start..self.pos]).into_owned();
        // `ReadToken` consumes the single delimiting space.
        if self.pos < self.buf.len() && self.buf[self.pos] == b' ' {
            self.pos += 1;
        }
        Ok(tok)
    }

    /// Kaldi `ExpectToken`.
    pub fn expect(&mut self, expected: &str) -> Result<()> {
        let offset = self.pos;
        let found = self.token()?;
        if found != expected {
            return Err(KaldiReadError::BadToken {
                offset,
                expected: expected.to_string(),
                found,
            });
        }
        Ok(())
    }

    /// Read a token only if it matches; otherwise leave the cursor untouched.
    pub fn accept(&mut self, expected: &str) -> Result<bool> {
        let save = self.pos;
        match self.token() {
            Ok(t) if t == expected => Ok(true),
            _ => {
                self.pos = save;
                Ok(false)
            }
        }
    }

    /// Kaldi `ReadBasicType<int32>`: a size byte of 4, then the value.
    pub fn i32(&mut self) -> Result<i32> {
        let offset = self.pos;
        let sz = self.byte()?;
        if sz != 4 {
            return Err(KaldiReadError::BadWidth {
                offset,
                what: "integer",
                want: 4,
                got: sz,
            });
        }
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Kaldi `ReadBasicType<uint32>`. Unsigned types are marked by a *negative*
    /// size byte (`io-funcs-inl.h:39`: `len_c = is_signed ? 1 : -1 * sizeof(T)`),
    /// so a `uint32` is announced as `-4`, i.e. the byte `0xfc`.
    pub fn u32(&mut self) -> Result<u32> {
        let offset = self.pos;
        let sz = self.byte()?;
        if sz as i8 != -4 {
            return Err(KaldiReadError::BadWidth {
                offset,
                what: "unsigned integer",
                want: 4,
                got: sz,
            });
        }
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Kaldi `ReadBasicType<BaseFloat>`. Accepts either width and widens.
    pub fn float(&mut self) -> Result<f32> {
        let offset = self.pos;
        let sz = self.byte()?;
        match sz {
            4 => {
                let b = self.take(4)?;
                Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            8 => {
                let b = self.take(8)?;
                let mut a = [0u8; 8];
                a.copy_from_slice(b);
                Ok(f64::from_le_bytes(a) as f32)
            }
            got => Err(KaldiReadError::BadWidth {
                offset,
                what: "float",
                want: 4,
                got,
            }),
        }
    }

    /// Kaldi `ReadIntegerVector<int32>`: element-size byte, raw `int32` count,
    /// then packed elements. Note the count has no size byte of its own.
    pub fn int_vec(&mut self) -> Result<Vec<i32>> {
        let offset = self.pos;
        let sz = self.byte()?;
        if sz != 4 {
            return Err(KaldiReadError::BadWidth {
                offset,
                what: "integer-vector element",
                want: 4,
                got: sz,
            });
        }
        let b = self.take(4)?;
        let n = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        if n < 0 {
            return Err(KaldiReadError::Malformed {
                offset,
                msg: format!("negative integer-vector length {n}"),
            });
        }
        let data = self.take(4 * n as usize)?;
        Ok(data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Kaldi `Vector<float>::Read`: an `FV`/`DV` token, a dimension, raw data.
    pub fn vector(&mut self) -> Result<Vec<f32>> {
        let offset = self.pos;
        let tok = self.token()?;
        let width = match tok.as_str() {
            "FV" => 4usize,
            "DV" => 8usize,
            _ => {
                return Err(KaldiReadError::BadToken {
                    offset,
                    expected: "FV or DV".into(),
                    found: tok,
                });
            }
        };
        let n = self.i32()?;
        if n < 0 {
            return Err(KaldiReadError::Malformed {
                offset,
                msg: format!("negative vector dim {n}"),
            });
        }
        let data = self.take(width * n as usize)?;
        Ok(decode_floats(data, width))
    }

    /// Kaldi `Matrix<float>::Read`: an `FM`/`DM` token, rows, cols, raw
    /// row-major data. Returns `(rows, cols, data)`.
    pub fn matrix(&mut self) -> Result<(usize, usize, Vec<f32>)> {
        let offset = self.pos;
        let tok = self.token()?;
        let width = match tok.as_str() {
            "FM" => 4usize,
            "DM" => 8usize,
            _ => {
                return Err(KaldiReadError::BadToken {
                    offset,
                    expected: "FM or DM".into(),
                    found: tok,
                });
            }
        };
        let rows = self.i32()?;
        let cols = self.i32()?;
        if rows < 0 || cols < 0 {
            return Err(KaldiReadError::Malformed {
                offset,
                msg: format!("negative matrix dims {rows}x{cols}"),
            });
        }
        let (rows, cols) = (rows as usize, cols as usize);
        let data = self.take(width * rows * cols)?;
        Ok((rows, cols, decode_floats(data, width)))
    }

    pub fn malformed(&self, msg: impl Into<String>) -> KaldiReadError {
        KaldiReadError::Malformed {
            offset: self.pos,
            msg: msg.into(),
        }
    }
}

fn decode_floats(data: &[u8], width: usize) -> Vec<f32> {
    if width == 4 {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        data.chunks_exact(8)
            .map(|c| {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                f64::from_le_bytes(a) as f32
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the bytes Kaldi would write, so the reader is tested against the
    /// writer's layout rather than against itself.
    fn write_token(out: &mut Vec<u8>, t: &str) {
        out.extend_from_slice(t.as_bytes());
        out.push(b' ');
    }
    fn write_i32(out: &mut Vec<u8>, v: i32) {
        out.push(4);
        out.extend_from_slice(&v.to_le_bytes());
    }

    #[test]
    fn rejects_a_file_without_the_binary_header() {
        assert!(matches!(
            KaldiReader::new(b"xx"),
            Err(KaldiReadError::NotBinary { .. })
        ));
    }

    #[test]
    fn reads_tokens_and_basic_types() {
        let mut b = vec![0u8, b'B'];
        write_token(&mut b, "<Hello>");
        write_i32(&mut b, -7);
        b.push(4);
        b.extend_from_slice(&1.5f32.to_le_bytes());

        let mut r = KaldiReader::new(&b).unwrap();
        r.expect("<Hello>").unwrap();
        assert_eq!(r.i32().unwrap(), -7);
        assert_eq!(r.float().unwrap(), 1.5);
        assert!(r.is_empty());
    }

    #[test]
    fn reads_an_integer_vector_whose_count_has_no_size_byte() {
        let mut b = vec![0u8, b'B'];
        b.push(4); // element size
        b.extend_from_slice(&3i32.to_le_bytes()); // raw count
        for v in [1i32, -2, 300] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let mut r = KaldiReader::new(&b).unwrap();
        assert_eq!(r.int_vec().unwrap(), vec![1, -2, 300]);
    }

    #[test]
    fn reads_a_float_vector_and_matrix() {
        let mut b = vec![0u8, b'B'];
        write_token(&mut b, "FV");
        write_i32(&mut b, 2);
        for v in [0.25f32, -4.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        write_token(&mut b, "FM");
        write_i32(&mut b, 2);
        write_i32(&mut b, 3);
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }

        let mut r = KaldiReader::new(&b).unwrap();
        assert_eq!(r.vector().unwrap(), vec![0.25, -4.0]);
        let (rows, cols, data) = r.matrix().unwrap();
        assert_eq!((rows, cols), (2, 3));
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn a_double_matrix_widens_to_f32() {
        let mut b = vec![0u8, b'B'];
        write_token(&mut b, "DM");
        write_i32(&mut b, 1);
        write_i32(&mut b, 2);
        for v in [1.5f64, -0.5] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let mut r = KaldiReader::new(&b).unwrap();
        let (_, _, data) = r.matrix().unwrap();
        assert_eq!(data, vec![1.5f32, -0.5]);
    }

    #[test]
    fn a_wrong_token_reports_where_it_happened() {
        let mut b = vec![0u8, b'B'];
        write_token(&mut b, "<Nope>");
        let mut r = KaldiReader::new(&b).unwrap();
        let err = r.expect("<Yes>").unwrap_err();
        assert!(matches!(err, KaldiReadError::BadToken { .. }));
    }

    #[test]
    fn accept_leaves_the_cursor_alone_on_a_miss() {
        let mut b = vec![0u8, b'B'];
        write_token(&mut b, "<A>");
        let mut r = KaldiReader::new(&b).unwrap();
        assert!(!r.accept("<B>").unwrap());
        assert!(r.accept("<A>").unwrap());
    }
}
