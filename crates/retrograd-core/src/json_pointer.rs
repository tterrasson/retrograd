//! JSON Pointer construction shared by the declarative and HTTP layers.

/// A JSON Pointer whose segment lifetime is scoped by [`Self::with_segment`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PointerPath(String);

impl PointerPath {
    pub fn new(root: impl Into<String>) -> Self {
        Self(root.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Visits one child and restores the parent even when `visit` returns an
    /// error through `?`.
    pub fn with_segment<T>(
        &mut self,
        segment: impl AsRef<str>,
        visit: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let restore = self.0.len();
        self.0.push('/');
        escape_into(segment.as_ref(), &mut self.0);
        let result = visit(self);
        self.0.truncate(restore);
        result
    }
}

impl std::fmt::Display for PointerPath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Escapes one RFC 6901 reference token.
pub fn escape_json_pointer(segment: &str) -> String {
    let mut escaped = String::with_capacity(segment.len());
    escape_into(segment, &mut escaped);
    escaped
}

fn escape_into(segment: &str, output: &mut String) {
    for character in segment.chars() {
        match character {
            '~' => output.push_str("~0"),
            '/' => output.push_str("~1"),
            other => output.push(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_are_escaped_and_always_popped() {
        let mut path = PointerPath::new("/params");
        let seen = path.with_segment("a/b~c", |path| path.to_string());
        assert_eq!(seen, "/params/a~1b~0c");
        assert_eq!(path.as_str(), "/params");
    }
}
