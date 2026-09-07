//! Audio decoding, resampling and WAV writing.
//!
//! Decodes wav/flac/mp3 with symphonia, downmixes to mono by averaging channels,
//! and resamples with rubato's asynchronous sinc resampler.
//!
//! Samples are kept in the `-1..1` float convention throughout. Kaldi reads WAV
//! files as raw `int16` values; the `feat` module multiplies by 32768 before MFCC
//! to match Kaldi's numeric scale. That scaling deliberately does *not* live here.

use std::fs::File;
use std::path::Path;

use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction, audioadapter_buffers::direct::SequentialSliceOfVecs,
};
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

/// Mono waveform with samples in `-1..1`.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Audio {
    /// Duration in seconds.
    pub fn duration_s(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("failed to open audio file {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to decode audio file {path}: {source}")]
    Decode {
        path: String,
        #[source]
        source: SymphoniaError,
    },
    #[error("no audio track found in {0}")]
    NoAudioTrack(String),
    #[error("audio track in {0} has no codec parameters")]
    NoCodecParams(String),
    #[error("could not determine the sample rate of {0}")]
    UnknownSampleRate(String),
    #[error("failed to write wav file {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: hound::Error,
    },
}

/// Read an audio file, downmixing to a single mono channel by averaging.
///
/// Supports every container/codec enabled in the workspace `symphonia` features
/// (wav, flac, mp3, plus the pcm/adpcm codecs used inside wav).
pub fn read(path: &Path) -> Result<Audio, AudioError> {
    let disp = path.display().to_string();
    let file = File::open(path).map_err(|source| AudioError::Open {
        path: disp.clone(),
        source,
    })?;

    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|source| AudioError::Decode {
            path: disp.clone(),
            source,
        })?;

    let track = format
        .first_track_known_codec(TrackType::Audio)
        .ok_or_else(|| AudioError::NoAudioTrack(disp.clone()))?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| AudioError::NoCodecParams(disp.clone()))?
        .clone();

    // Sample rate may be declared on the track; otherwise we take it from the
    // first decoded buffer's spec.
    let mut sample_rate = audio_params.sample_rate;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())
        .map_err(|source| AudioError::Decode {
            path: disp.clone(),
            source,
        })?;

    let mut samples: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            // A truncated final packet is common in the wild; keep what we decoded.
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(source) => {
                return Err(AudioError::Decode { path: disp, source });
            }
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded: GenericAudioBufferRef<'_> = match decoder.decode(&packet) {
            Ok(d) => d,
            // Recoverable per-packet errors: skip the packet, as symphonia advises.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(source) => {
                return Err(AudioError::Decode { path: disp, source });
            }
        };

        let spec = decoded.spec();
        if sample_rate.is_none() {
            sample_rate = Some(spec.rate());
        }
        let channels = spec.channels().count().max(1);
        let frames = decoded.frames();
        if frames == 0 {
            continue;
        }

        interleaved.clear();
        interleaved.resize(frames * channels, 0.0f32);
        decoded.copy_to_slice_interleaved(&mut interleaved[..]);

        samples.reserve(frames);
        if channels == 1 {
            samples.extend_from_slice(&interleaved);
        } else {
            let scale = 1.0f32 / channels as f32;
            for frame in interleaved.chunks_exact(channels) {
                let sum: f32 = frame.iter().sum();
                samples.push(sum * scale);
            }
        }
    }

    let sample_rate = sample_rate.ok_or(AudioError::UnknownSampleRate(disp))?;

    Ok(Audio {
        samples,
        sample_rate,
    })
}

/// Sinc resampling to `target_rate`. Identity (a clone) if the rate already matches.
///
/// CONTRACT-DEVIATION: the contract names rubato's `SincFixedIn`. rubato 5.0.0
/// removed that type; the equivalent is `Async::new_sinc(.., FixedAsync::Input)`,
/// which is what this uses. Behaviour is the same fixed-input-size sinc resampler.
pub fn resample(a: &Audio, target_rate: u32) -> Audio {
    if a.sample_rate == target_rate || a.samples.is_empty() || a.sample_rate == 0 {
        return Audio {
            samples: a.samples.clone(),
            sample_rate: if a.sample_rate == 0 {
                target_rate
            } else {
                a.sample_rate
            },
        };
    }

    let ratio = target_rate as f64 / a.sample_rate as f64;
    // sinc_len 256 with a BlackmanHarris2 window is rubato's recommended
    // high-quality setting; the cutoff is then derived automatically. MFA uses
    // librosa's soxr_hq; the two differ only in the anti-alias transition band
    // above 7.5 kHz (see plans/parity/parity_004_feats.py).
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: None,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    };

    let chunk_size = 1024usize;
    let mut resampler =
        match Async::<f32>::new_sinc(ratio, 1.0, &params, chunk_size, 1, FixedAsync::Input) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("failed to construct resampler: {e}; returning input unchanged");
                return a.clone();
            }
        };

    let input_len = a.samples.len();
    let input_data = vec![a.samples.clone()];
    let input = match SequentialSliceOfVecs::new(&input_data, 1, input_len) {
        Ok(i) => i,
        Err(e) => {
            tracing::error!("failed to wrap resampler input: {e}");
            return a.clone();
        }
    };

    let needed = resampler.process_all_needed_output_len(input_len);
    let mut output_data = vec![vec![0.0f32; needed]];
    let mut output = match SequentialSliceOfVecs::new_mut(&mut output_data, 1, needed) {
        Ok(o) => o,
        Err(e) => {
            tracing::error!("failed to wrap resampler output: {e}");
            return a.clone();
        }
    };

    let out_len = match resampler.process_all_into_buffer(&input, &mut output, input_len, None) {
        Ok((_in_len, out_len)) => out_len,
        Err(e) => {
            tracing::error!("resampling failed: {e}; returning input unchanged");
            return a.clone();
        }
    };

    let mut samples = output_data.pop().expect("one channel of output");
    samples.truncate(out_len);

    Audio {
        samples,
        sample_rate: target_rate,
    }
}

/// Read and resample to 16 kHz, the rate every model in this project uses.
pub fn read_16k(path: &Path) -> Result<Audio, AudioError> {
    let a = read(path)?;
    let mut r = resample(&a, 16_000);
    // kalpy hands Kaldi integer-valued samples (`np.round(wave * 32768)`), so a
    // resampled float waveform is snapped to the int16 grid to match.
    for s in &mut r.samples {
        *s = (*s * 32768.0).round() / 32768.0;
    }
    Ok(r)
}

/// Write 16-bit PCM mono WAV.
pub fn write_wav(path: &Path, a: &Audio) -> Result<(), AudioError> {
    let disp = path.display().to_string();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: a.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|source| AudioError::Write {
        path: disp.clone(),
        source,
    })?;
    for &s in &a.samples {
        // Same convention as the read path: -1..1 maps onto the int16 range with
        // 32768 as the scale, clamped so that +1.0 does not wrap to -32768.
        let v = (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(v).map_err(|source| AudioError::Write {
            path: disp.clone(),
            source,
        })?;
    }
    writer
        .finalize()
        .map_err(|source| AudioError::Write { path: disp, source })
}

/// Extract `[start_s, end_s)` as a new `Audio`. Out-of-range bounds are clamped;
/// an inverted or empty range yields an empty waveform at the same rate.
pub fn slice(a: &Audio, start_s: f32, end_s: f32) -> Audio {
    let n = a.samples.len();
    let rate = a.sample_rate as f64;
    let start = ((start_s.max(0.0) as f64) * rate).round() as usize;
    let end = ((end_s.max(0.0) as f64) * rate).round() as usize;
    let start = start.min(n);
    let end = end.min(n);
    let samples = if end > start {
        a.samples[start..end].to_vec()
    } else {
        Vec::new()
    };
    Audio {
        samples,
        sample_rate: a.sample_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, secs: f32, freq: f32) -> Audio {
        let n = (rate as f32 * secs) as usize;
        let samples = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin() * 0.5)
            .collect();
        Audio {
            samples,
            sample_rate: rate,
        }
    }

    #[test]
    fn resample_same_rate_is_identity() {
        let a = tone(16_000, 0.1, 440.0);
        let b = resample(&a, 16_000);
        assert_eq!(a.samples, b.samples);
        assert_eq!(b.sample_rate, 16_000);
    }

    #[test]
    fn resample_changes_length_by_ratio() {
        let a = tone(44_100, 0.5, 440.0);
        let b = resample(&a, 16_000);
        assert_eq!(b.sample_rate, 16_000);
        let expected = a.samples.len() as f64 * 16_000.0 / 44_100.0;
        let diff = (b.samples.len() as f64 - expected).abs();
        // Allow a small edge tolerance from the resampler's chunking.
        assert!(
            diff < 64.0,
            "len {} vs expected {expected}",
            b.samples.len()
        );
    }

    #[test]
    fn resample_preserves_a_sine_in_band() {
        // 440 Hz is far below both Nyquist limits, so it must survive resampling.
        let a = tone(48_000, 0.25, 440.0);
        let b = resample(&a, 16_000);
        // Skip the filter's transient at both ends.
        let skip = 400;
        assert!(b.samples.len() > 2 * skip);
        let peak = b.samples[skip..b.samples.len() - skip]
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.5).abs() < 0.05, "peak {peak}");
    }

    #[test]
    fn slice_clamps_and_extracts() {
        let a = Audio {
            samples: (0..100).map(|i| i as f32).collect(),
            sample_rate: 100,
        };
        let s = slice(&a, 0.1, 0.2);
        assert_eq!(s.samples.len(), 10);
        assert_eq!(s.samples[0], 10.0);
        assert_eq!(s.sample_rate, 100);

        // Past the end clamps.
        let s = slice(&a, 0.9, 5.0);
        assert_eq!(s.samples.len(), 10);

        // Inverted range is empty.
        assert!(slice(&a, 0.5, 0.2).samples.is_empty());
    }

    #[test]
    fn wav_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("viter_audio_roundtrip_test.wav");
        let a = tone(16_000, 0.05, 1000.0);
        write_wav(&path, &a).expect("write");
        let b = read(&path).expect("read");
        assert_eq!(b.sample_rate, 16_000);
        assert_eq!(b.samples.len(), a.samples.len());
        for (x, y) in a.samples.iter().zip(b.samples.iter()) {
            // 16-bit quantisation error.
            assert!((x - y).abs() < 1.0 / 32768.0 + 1e-7, "{x} vs {y}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn duration_of_known_length() {
        let a = tone(16_000, 1.0, 100.0);
        assert!((a.duration_s() - 1.0).abs() < 1e-6);
        assert!(!a.is_empty());
    }
}
