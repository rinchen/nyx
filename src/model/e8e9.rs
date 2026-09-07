//! E8E9 transform for x86/AMD64 executables.
//!
//! In x86 machine code, the E8 (call) and E9 (jmp near) instructions encode
//! a 32-bit **relative** offset. These offsets are dispersed throughout the
//! address space and hard to compress. The E8E9 transform converts them to
//! **absolute** offsets by adding the instruction's address, which groups
//! similar values together.
//!
//! Layout of an E8/E9 instruction: [E8/E9][disp32: little-endian]
//!
//! Transform: `disp32' = disp32 + instruction_address + 5`
//! Inverse:   `disp32 = disp32' - instruction_address - 5`
//!
//! Only E8 and E9 with 32-bit displacements are transformed. Other uses of
//! these opcodes (e.g., E8 as a prefix in newer CPUID, E9 as a padding byte)
//! are left unchanged to avoid corrupting non-code data.

/// E8E9 opcode bytes.
const E8: u8 = 0xE8;
const E9: u8 = 0xE9;

/// Transform an x86/AMD64 executable, converting relative offsets to absolute.
/// Returns the transformed data (same length as input).
#[must_use]
pub fn e8e9_transform(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0usize;
    while i < data.len() {
        if i + 5 <= data.len() && (data[i] == E8 || data[i] == E9) {
            // Read the 32-bit little-endian displacement
            let disp = u32::from_le_bytes([data[i + 1], data[i + 2], data[i + 3], data[i + 4]]);
            // Compute absolute offset: instruction_address + displacement + 5 (instruction length)
            let abs = disp.wrapping_add((i as u32).wrapping_add(5));
            // Emit opcode + absolute offset
            out.push(data[i]);
            out.extend_from_slice(&abs.to_le_bytes());
            i += 5;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

/// Inverse of [`e8e9_transform`], converting absolute offsets back to relative.
#[must_use]
pub fn e8e9_inverse(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0usize;
    while i < data.len() {
        if i + 5 <= data.len() && (data[i] == E8 || data[i] == E9) {
            // Read the 32-bit absolute offset
            let abs = u32::from_le_bytes([data[i + 1], data[i + 2], data[i + 3], data[i + 4]]);
            // Compute relative offset: absolute - instruction_address - 5
            let rel = abs.wrapping_sub((i as u32).wrapping_add(5));
            // Emit opcode + relative offset
            out.push(data[i]);
            out.extend_from_slice(&rel.to_le_bytes());
            i += 5;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_simple() {
        let code = vec![
            0x55,             // push rbp
            0x48, 0x89, 0xE5, // mov rbp, rsp
            0xE8, 0x10, 0x00, 0x00, 0x00, // call $+20 (relative offset 0x10)
            0xE9, 0x00, 0x00, 0x00, 0x00, // jmp $+5 (relative offset 0)
            0xC3, // ret
        ];
        let transformed = e8e9_transform(&code);
        let inverse = e8e9_inverse(&transformed);
        assert_eq!(inverse, code);
    }

    #[test]
    fn preserves_length() {
        let code = vec![0x55; 100];
        let transformed = e8e9_transform(&code);
        assert_eq!(transformed.len(), code.len());
    }

    #[test]
    fn transforms_call_instructions() {
        // E8 with displacement 0 at position 0
        let code = vec![0xE8, 0x00, 0x00, 0x00, 0x00];
        let transformed = e8e9_transform(&code);
        // After transform: E8 + (0 + 0 + 5) = E8 + 5 = [E8, 0x05, 0x00, 0x00, 0x00]
        assert_eq!(transformed[0], E8);
        let abs = u32::from_le_bytes([transformed[1], transformed[2], transformed[3], transformed[4]]);
        assert_eq!(abs, 5);
    }

    #[test]
    fn round_trip_large_displacement() {
        // E8 at position 1000 with displacement that would overflow if not using wrapping
        let mut code = vec![0x90; 1000]; // NOP padding
        code.push(0xE8);
        code.extend_from_slice(&0x7FFFFFFFu32.to_le_bytes()); // Max positive displacement
        let transformed = e8e9_transform(&code);
        let inverse = e8e9_inverse(&transformed);
        assert_eq!(inverse, code);
    }
}
