//! Declares a closed vocabulary once.

/// Declares a fieldless enum together with the string each variant is spelled
/// as, and derives from that single list both `ALL` and `as_str()`.
///
/// This keeps four workspace vocabularies from being written repeatedly as
/// variants, a `const ALL`,
/// enumerating them again, an `as_str()` restating what `#[serde(rename_all)]`
/// already produced. Tests kept the copies aligned, which worked; but a test
/// that exists only to check a file against itself is a cost every reader pays
/// and a gap that only closes after the fact.
///
/// Two forms, because the strings do not always mean the same thing:
///
/// * `enum Name: serde { … }` - the enum is serialized, and each variant gets
///   `#[serde(rename = …)]` from the declared spelling. Serde's wire form and
///   `as_str()` are then the same token in the source: they cannot disagree.
///   The caller still writes its own `derive`, so which of `Serialize` and
///   `Deserialize` it wants stays its decision.
/// * `enum Name { … }` - the strings are labels (a message, a URI slug) and
///   nothing serializes the enum. No serde attribute is emitted, so the macro
///   forces no dependency on a type that has no wire form.
///
/// ```
/// retrograd_core::wire_enum! {
///     /// What a record failed on.
///     #[derive(Clone, Copy, Debug, PartialEq, Eq)]
///     pub enum Failure {
///         /// Nothing but whitespace.
///         Empty = "empty",
///         Malformed = "malformed",
///     }
/// }
/// assert_eq!(Failure::Empty.as_str(), "empty");
/// assert_eq!(Failure::ALL.len(), 2);
/// ```
#[macro_export]
macro_rules! wire_enum {
    (
        $(#[$enum_meta:meta])*
        $visibility:vis enum $name:ident : serde {
            $( $(#[$variant_meta:meta])* $variant:ident = $wire:literal ),* $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        $visibility enum $name {
            $(
                $(#[$variant_meta])*
                #[serde(rename = $wire)]
                $variant,
            )*
        }

        $crate::wire_enum!(@accessors $name { $( $variant = $wire, )* });
    };

    (
        $(#[$enum_meta:meta])*
        $visibility:vis enum $name:ident {
            $( $(#[$variant_meta:meta])* $variant:ident = $wire:literal ),* $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        $visibility enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )*
        }

        $crate::wire_enum!(@accessors $name { $( $variant = $wire, )* });
    };

    (@accessors $name:ident { $( $variant:ident = $wire:literal, )* }) => {
        impl $name {
            /// Every variant, in declaration order.
            pub const ALL: &'static [$name] = &[ $( $name::$variant, )* ];

            /// The spelling declared beside the variant.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $( $name::$variant => $wire, )*
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    wire_enum! {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        pub enum Colour: serde {
            /// A doc comment on a variant survives the expansion.
            DeepBlue = "deep-blue",
            Red = "red",
        }
    }

    wire_enum! {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Label {
            Empty = "empty_record",
            Malformed = "invalid_json",
        }
    }

    #[test]
    fn the_declared_spelling_is_the_one_serde_writes() {
        for colour in Colour::ALL {
            assert_eq!(
                serde_json::to_string(colour).expect("a fieldless enum serializes"),
                format!("\"{}\"", colour.as_str()),
            );
        }
    }

    #[test]
    fn a_label_enum_needs_no_serde_impl_at_all() {
        assert_eq!(Label::ALL, &[Label::Empty, Label::Malformed]);
        assert_eq!(Label::Malformed.as_str(), "invalid_json");
    }
}
