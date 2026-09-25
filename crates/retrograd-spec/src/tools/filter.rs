//! The one name-pattern language used by every tool source.

/// `*` matches any run of characters; every other character is literal.
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let parts = pattern.split('*').collect::<Vec<_>>();
    if parts.len() == 1 {
        return pattern == name;
    }
    if !name.starts_with(parts[0]) || !name.ends_with(parts[parts.len() - 1]) {
        return false;
    }
    let mut cursor = parts[0].len();
    let middle_end = parts.len() - 1;
    for part in &parts[1..middle_end] {
        if part.is_empty() {
            continue;
        }
        let Some(found) = name[cursor..].find(part) else {
            return false;
        };
        cursor += found + part.len();
    }
    cursor <= name.len() - parts[parts.len() - 1].len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcard_language_is_stable() {
        for (pattern, name, expected) in [
            ("read_*", "read_file", true),
            ("*file", "read_file", true),
            ("r*d*f*", "read_file", true),
            ("bash", "bash", true),
            ("bash", "bashful", false),
            ("a*b", "ac", false),
        ] {
            assert_eq!(glob_match(pattern, name), expected, "{pattern} / {name}");
        }
    }
}
