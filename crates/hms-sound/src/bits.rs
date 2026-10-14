//! LSB-first bit reader, the bit packing Vorbis uses.
//!
//! Every read is bounds-checked and returns `None` past the end, so a
//! truncated or hostile packet ends a parse instead of panicking.

pub struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize, // in bits
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn seek_bits(&mut self, bits: usize) {
        self.pos = bits;
    }

    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    pub fn bits_left(&self) -> usize {
        self.buf.len().saturating_mul(8).saturating_sub(self.pos)
    }

    /// Read `n` bits (n <= 32), least-significant bit first.
    pub fn read(&mut self, n: u32) -> Option<u32> {
        debug_assert!(n <= 32);
        if self.bits_left() < n as usize {
            return None;
        }
        let mut v: u32 = 0;
        for i in 0..n {
            let byte = *self.buf.get(self.pos >> 3)?;
            v |= (((byte >> (self.pos & 7)) & 1) as u32) << i;
            self.pos += 1;
        }
        Some(v)
    }

    /// Read and discard `n` bits, where `n` may exceed 32.
    pub fn skip(&mut self, n: usize) -> Option<()> {
        if self.bits_left() < n {
            return None;
        }
        self.pos += n;
        Some(())
    }
}

/// `ilog` as Vorbis defines it: the number of bits needed to hold `x`.
pub fn ilog(mut x: u32) -> u32 {
    let mut n = 0;
    while x > 0 {
        n += 1;
        x >>= 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_lsb_first_and_stops_at_the_end() {
        let mut r = BitReader::new(&[0b1011_0001, 0xFF]);
        assert_eq!(r.read(1), Some(1));
        assert_eq!(r.read(4), Some(0b1000));
        assert_eq!(r.read(3), Some(0b101));
        assert_eq!(r.read(8), Some(0xFF));
        // Nothing left: every further read fails rather than panicking.
        assert_eq!(r.read(1), None);
        assert_eq!(r.bits_left(), 0);
    }

    #[test]
    fn an_oversized_read_fails_instead_of_wrapping() {
        let mut r = BitReader::new(&[0x01]);
        assert_eq!(r.read(9), None);
        // The position is untouched by a failed read.
        assert_eq!(r.bit_pos(), 0);
    }

    #[test]
    fn ilog_matches_the_spec() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(255), 8);
    }
}
