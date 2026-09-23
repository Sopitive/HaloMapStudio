//! Signed numeric text entry: a MINUS anywhere in the string is a sign flip.
//!
//! People type an angle and only then decide it should go the other way -- "90 -" has to mean
//! the same as "-90", because reaching back to the front of the field to insert the sign is
//! exactly the fiddly thing a Forger does not want to do mid-edit. This is the one parser every
//! signed field in the app uses (`egui::DragValue::custom_parser`), so the rule cannot drift
//! between the rotation fields, the position fields and the modal transform's typed entry.
//!
//! The rule:
//! * every `-` is a sign TOGGLE, wherever it sits (an odd count = negative, so `--90` = +90);
//!   the unicode minus / en dash / em dash count too, because that is what some keyboards and
//!   pasted text produce;
//! * whitespace around the number, the signs and the unit is ignored;
//! * a trailing `deg` / `degs` / `degree` / `degrees` / `°` is ignored (case-insensitive), so a
//!   field that shows `90°` can be retyped verbatim;
//! * whatever is left must be a plain unsigned number -- so `9 0` and `9-0` are REJECTED rather
//!   than silently read as 90 or 9;
//! * nothing parsable (empty, `-` alone, `.`) returns `None`, which `DragValue` treats as
//!   "leave the value alone".

/// Characters people actually produce when they mean "minus".
const MINUS: [char; 4] = ['-', '\u{2212}', '\u{2013}', '\u{2014}'];

/// Unit suffixes a number may carry, longest first (so `degrees` is not read as `deg` + `rees`).
const UNITS: [&str; 6] = ["degrees", "degree", "degs", "deg", "°", "\u{00BA}"];

fn strip_unit(s: &str) -> Option<&str> {
    let lower = s.to_ascii_lowercase();
    for u in UNITS {
        if lower.len() >= u.len() && lower.ends_with(u) {
            return Some(&s[..s.len() - u.len()]);
        }
    }
    None
}

/// Parse a field where a negative value is meaningful, accepting the sign anywhere.
/// `None` = not a number (the caller keeps the current value).
pub fn parse_signed(text: &str) -> Option<f64> {
    let mut s = text.trim();
    let mut neg = false;
    // Peel signs / units / whitespace off BOTH ends until nothing more comes off. What is left
    // is the bare magnitude -- anything else (an interior space or sign) makes it unparsable.
    loop {
        let before = s;
        s = s.trim();
        if let Some(rest) = strip_unit(s) {
            s = rest;
        }
        for m in MINUS {
            if let Some(rest) = s.strip_prefix(m) {
                neg = !neg;
                s = rest;
            }
            if let Some(rest) = s.strip_suffix(m) {
                neg = !neg;
                s = rest;
            }
        }
        s = s.strip_prefix('+').unwrap_or(s);
        s = s.strip_suffix('+').unwrap_or(s);
        if s == before {
            break;
        }
    }
    if s.is_empty() || !s.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }
    let v: f64 = s.parse().ok()?;
    if !v.is_finite() {
        return None;
    }
    Some(if neg { -v } else { v })
}

#[cfg(test)]
mod tests {
    use super::parse_signed as p;

    #[test]
    fn trailing_minus_is_a_sign_flip() {
        assert_eq!(p("90 -"), Some(-90.0));
        assert_eq!(p("90-"), Some(-90.0));
        assert_eq!(p("-90"), Some(-90.0));
        assert_eq!(p("- 90"), Some(-90.0));
        assert_eq!(p("90"), Some(90.0));
    }

    #[test]
    fn an_even_count_of_minuses_is_positive() {
        assert_eq!(p("--90"), Some(90.0));
        assert_eq!(p("-90-"), Some(90.0));
        assert_eq!(p("---90"), Some(-90.0));
    }

    #[test]
    fn units_and_whitespace_are_tolerated() {
        assert_eq!(p("  90  "), Some(90.0));
        assert_eq!(p("90deg"), Some(90.0));
        assert_eq!(p("90 DEG -"), Some(-90.0));
        assert_eq!(p("90degrees"), Some(90.0));
        assert_eq!(p("90°"), Some(90.0));
        assert_eq!(p("-90°"), Some(-90.0));
        assert_eq!(p("90° -"), Some(-90.0));
        assert_eq!(p("12.5-"), Some(-12.5));
        assert_eq!(p("+90"), Some(90.0));
    }

    #[test]
    fn a_unicode_minus_counts() {
        assert_eq!(p("90\u{2212}"), Some(-90.0)); // U+2212 MINUS SIGN
        assert_eq!(p("\u{2013}90"), Some(-90.0)); // en dash
    }

    #[test]
    fn nonsense_is_rejected_not_guessed() {
        assert_eq!(p("9 0"), None); // an interior space is a typo, not a thousands separator
        assert_eq!(p("9-0"), None); // an interior sign is not a flip
        assert_eq!(p(""), None);
        assert_eq!(p("   "), None);
        assert_eq!(p("-"), None);
        assert_eq!(p("."), None);
        assert_eq!(p("deg"), None);
        assert_eq!(p("abc"), None);
        assert_eq!(p("1e3"), None);
        assert_eq!(p("0x10"), None);
    }

    #[test]
    fn plain_values_round_trip_like_the_default_parser() {
        assert_eq!(p("0"), Some(0.0));
        assert_eq!(p("0.5"), Some(0.5));
        assert_eq!(p(".5"), Some(0.5));
        assert_eq!(p("360"), Some(360.0));
        assert_eq!(p("-0.25"), Some(-0.25));
    }
}
