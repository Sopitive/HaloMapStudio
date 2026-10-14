//! A minimal Ogg writer, enough to wrap recovered Vorbis packets.
//!
//! This exists so one bank sample can be handed to *any* Vorbis decoder, and
//! so a sample can be exported as a plain `.ogg` file. It is not a general
//! Ogg muxer: it writes a single logical stream and at most 255 segments per
//! page, which is all the page builder here needs.

/// Ogg's CRC-32: polynomial 0x04c11db7, no reflection, no initial or final
/// inversion. Not the same as zlib's crc32.
fn crc32_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    for (i, slot) in t.iter_mut().enumerate() {
        let mut r = (i as u32) << 24;
        for _ in 0..8 {
            r = if r & 0x8000_0000 != 0 {
                (r << 1) ^ 0x04c1_1db7
            } else {
                r << 1
            };
        }
        *slot = r;
    }
    t
}

pub fn ogg_crc(bytes: &[u8]) -> u32 {
    let t = crc32_table();
    let mut r: u32 = 0;
    for b in bytes {
        r = (r << 8) ^ t[(((r >> 24) as u8) ^ *b) as usize];
    }
    r
}

pub const BOS: u8 = 0x02;
pub const EOS: u8 = 0x04;
pub const CONTINUED: u8 = 0x01;

/// Maximum packets per page, chosen so the 255-segment limit cannot be hit by
/// packets of any size up to 64 KiB.
pub const PACKETS_PER_PAGE: usize = 16;

/// Build one Ogg page. Returns `None` if the packets need more than 255
/// lacing segments.
pub fn page(
    packets: &[&[u8]],
    serial: u32,
    sequence: u32,
    granule: i64,
    flags: u8,
) -> Option<Vec<u8>> {
    let mut segments: Vec<u8> = Vec::new();
    for p in packets {
        let mut n = p.len();
        while n >= 255 {
            segments.push(255);
            n -= 255;
        }
        segments.push(n as u8);
    }
    if segments.len() > 255 {
        return None;
    }
    let mut out = Vec::with_capacity(27 + segments.len() + packets.iter().map(|p| p.len()).sum::<usize>());
    out.extend_from_slice(b"OggS");
    out.push(0); // stream structure version
    out.push(flags);
    out.extend_from_slice(&granule.to_le_bytes());
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&sequence.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // checksum, filled in below
    out.push(segments.len() as u8);
    out.extend_from_slice(&segments);
    for p in packets {
        out.extend_from_slice(p);
    }
    let c = ogg_crc(&out);
    out[22..26].copy_from_slice(&c.to_le_bytes());
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_has_the_right_shape() {
        let p: &[u8] = &[1, 2, 3];
        let pg = page(&[p], 7, 0, 0, BOS).unwrap();
        assert_eq!(&pg[..4], b"OggS");
        assert_eq!(pg[4], 0);
        assert_eq!(pg[5], BOS);
        assert_eq!(u32::from_le_bytes(pg[14..18].try_into().unwrap()), 7);
        assert_eq!(pg[26], 1); // one segment
        assert_eq!(pg[27], 3); // of three bytes
        assert_eq!(&pg[28..], &[1, 2, 3]);
    }

    #[test]
    fn the_checksum_covers_the_page_with_its_own_field_zeroed() {
        let p: &[u8] = &[9; 10];
        let pg = page(&[p], 1, 2, 3, 0).unwrap();
        let stored = u32::from_le_bytes(pg[22..26].try_into().unwrap());
        let mut zeroed = pg.clone();
        zeroed[22..26].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(stored, ogg_crc(&zeroed));
        assert_ne!(stored, 0);
    }

    #[test]
    fn a_packet_of_exactly_255_bytes_gets_a_zero_terminator_segment() {
        let p = vec![0u8; 255];
        let pg = page(&[&p], 1, 0, 0, 0).unwrap();
        assert_eq!(pg[26], 2);
        assert_eq!(pg[27], 255);
        assert_eq!(pg[28], 0);
    }

    #[test]
    fn too_many_segments_is_refused_rather_than_truncating_the_table() {
        // 256 one-byte packets need 256 segments.
        let one = [0u8; 1];
        let refs: Vec<&[u8]> = (0..256).map(|_| one.as_slice()).collect();
        assert!(page(&refs, 1, 0, 0, 0).is_none());
    }
}
