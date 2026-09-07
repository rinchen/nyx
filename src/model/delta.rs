//! Delta/stride transforms for Binary blocks.
//!
//! Binary data often has strong byte-to-byte correlations where the current
//! byte is similar to the previous byte. A simple delta transform replaces
//! each byte with the difference from the previous byte, making the stream
//! more compressible.
//!
//! For bytes where the delta is small (typically within [-128, 127]), the
//! resulting values cluster around 0, which improves CM prediction.
//!
//! Additionally, a simple stride detector looks for repeating patterns
//! at fixed intervals and replaces them with (period, value) pairs.

/// Delta transform: replace each byte with `byte - prev_byte` (mod 256).
/// The first byte is kept unchanged.
///
/// Returns the transformed data (same length as input).
#[must_use]
pub fn delta_transform(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    out.push(data[0]);
    for i in 1..data.len() {
        // Signed subtraction, wrapped to u8
        out.push(data[i].wrapping_sub(data[i - 1]));
    }
    out
}

/// Inverse of [`delta_transform`], reconstructing the original bytes.
#[must_use]
pub fn delta_inverse(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    out.push(data[0]);
    let mut prev = data[0];
    for i in 1..data.len() {
        // Reverse: original = prev + delta (mod 256)
        let orig = prev.wrapping_add(data[i]);
        out.push(orig);
        prev = orig;
    }
    out
}

/// Simple stride detection for repeating patterns.
///
/// Scans for the most common stride period (1..=8) where
/// `data[i] == data[i - stride]` for multiple positions.
/// Returns `(best_stride, best_period_count)` or `(0, 0)` if no stride detected.
#[must_use]
pub fn detect_stride(data: &[u8]) -> (usize, usize) {
    if data.len() < 32 {
        return (0, 0);
    }

    let mut best_stride = 0usize;
    let mut best_count = 0usize;

    for stride in 1..=8 {
        let mut count = 0usize;
        for i in stride..data.len() {
            if data[i] == data[i - stride] {
                count += 1;
            }
        }
        if count > best_count && count > data.len() / 4 {
            best_count = count;
            best_stride = stride;
        }
    }

    (best_stride, best_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_simple() {
        let data = vec![10u8, 12, 15, 13, 20, 25, 22];
        let transformed = delta_transform(&data);
        let recovered = delta_inverse(&transformed);
        assert_eq!(recovered, data);
    }

    #[test]
    fn preserves_length() {
        let data = vec![0u8; 100];
        let transformed = delta_transform(&data);
        assert_eq!(transformed.len(), data.len());
    }

    #[test]
    fn all_zeros() {
        let data = vec![42u8; 50];
        let transformed = delta_transform(&data);
        // First byte stays, rest should be 0
        assert_eq!(transformed[0], 42);
        for &b in &transformed[1..] {
            assert_eq!(b, 0);
        }
        let recovered = delta_inverse(&transformed);
        assert_eq!(recovered, data);
    }

    #[test]
    fn detects_stride() {
        // Repeating pattern: ABABABAB...
        let data: Vec<u8> = (0..100).map(|i| if i % 2 == 0 { 1u8 } else { 2 }).collect();
        let (stride, count) = detect_stride(&data);
        assert_eq!(stride, 2);
        assert!(count > 40); // Most bytes match at stride 2
    }
}
