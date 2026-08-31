//! Self-describing container format for `nyx` (`NYX1`).
//!
//! Layout: `[MAGIC(4)][Header(7)][BlockEntry * num_blocks (13 each)][block payloads...]`.
//! Each block payload is preceded by its `BlockEntry` (compressed length, original length,
//! method, CRC32 of the *original* block).

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use std::io::{Cursor, Read};

pub const MAGIC: &[u8; 4] = b"NYX1";
pub const VERSION: u8 = 1;

/// Container header (7 bytes after the 4-byte magic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub version: u8,
    pub flags: u8,
    pub block_size_log: u8,
    pub num_blocks: u32,
}

/// Per-block record (13 bytes): tells the decoder how to find and validate the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockEntry {
    pub comp_len: u32,
    pub orig_len: u32,
    pub method: u8,
    pub crc32: u32,
}

impl Header {
    /// Serialize the header (including magic) onto `w`.
    ///
    /// # Panics
    ///
    /// Never panics in practice: writing to an in-memory `Vec` cannot fail. The `.unwrap()`
    /// is only to satisfy the `byteorder` `io::Result` contract.
    pub fn write(&self, w: &mut Vec<u8>) {
        w.extend_from_slice(MAGIC);
        w.push(self.version);
        w.push(self.flags);
        w.push(self.block_size_log);
        w.write_u32::<LittleEndian>(self.num_blocks).unwrap();
    }

    /// Parse a header (and consume the magic) from `r`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error on truncation or a bad magic.
    pub fn read(r: &mut Cursor<&[u8]>) -> std::io::Result<Self> {
        let mut m = [0u8; 4];
        r.read_exact(&mut m)?;
        if &m != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad nyx magic",
            ));
        }
        Ok(Self {
            version: r.read_u8()?,
            flags: r.read_u8()?,
            block_size_log: r.read_u8()?,
            num_blocks: r.read_u32::<LittleEndian>()?,
        })
    }
}

impl BlockEntry {
    /// Serialize this entry (13 bytes).
    ///
    /// # Panics
    ///
    /// Never panics in practice: writing to an in-memory `Vec` cannot fail.
    pub fn write(&self, w: &mut Vec<u8>) {
        w.write_u32::<LittleEndian>(self.comp_len).unwrap();
        w.write_u32::<LittleEndian>(self.orig_len).unwrap();
        w.push(self.method);
        w.write_u32::<LittleEndian>(self.crc32).unwrap();
    }

    /// Parse an entry from `r`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error on truncation.
    pub fn read(r: &mut Cursor<&[u8]>) -> std::io::Result<Self> {
        Ok(Self {
            comp_len: r.read_u32::<LittleEndian>()?,
            orig_len: r.read_u32::<LittleEndian>()?,
            method: r.read_u8()?,
            crc32: r.read_u32::<LittleEndian>()?,
        })
    }
}

/// CRC32 of `buf` (used to validate decompressed blocks against corruption).
#[must_use]
pub fn crc32(buf: &[u8]) -> u32 {
    let mut h = Hasher::new();
    h.update(buf);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_roundtrip() {
        let header = Header {
            version: VERSION,
            flags: 0,
            block_size_log: 16,
            num_blocks: 2,
        };
        let entries = [
            BlockEntry {
                comp_len: 10,
                orig_len: 100,
                method: 1,
                crc32: 0x1234_5678,
            },
            BlockEntry {
                comp_len: 20,
                orig_len: 200,
                method: 0,
                crc32: 0x9abc_def0,
            },
        ];
        let mut buf = Vec::new();
        header.write(&mut buf);
        for e in &entries {
            e.write(&mut buf);
        }
        let mut cur = Cursor::new(buf.as_slice());
        let got_h = Header::read(&mut cur).expect("header read");
        assert_eq!(got_h, header);
        let got_e0 = BlockEntry::read(&mut cur).expect("entry 0");
        let got_e1 = BlockEntry::read(&mut cur).expect("entry 1");
        assert_eq!(got_e0, entries[0]);
        assert_eq!(got_e1, entries[1]);
    }

    #[test]
    fn crc32_is_deterministic() {
        assert_eq!(crc32(b"nyx"), crc32(b"nyx"));
        assert_ne!(crc32(b"nyx"), crc32(b"xxx"));
    }

    #[test]
    fn bad_magic_errors() {
        let buf = b"XXXX\x01\x00\x10\x00\x00\x00\x00";
        let mut cur = Cursor::new(buf.as_slice());
        assert!(Header::read(&mut cur).is_err());
    }
}
