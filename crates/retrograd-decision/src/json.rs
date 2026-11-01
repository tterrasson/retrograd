//! A JSON value whose objects keep the order of their keys.
//!
//! The `systemone` template prints the state with `tojson`, and the model was
//! trained on the keys in the order the request wrote them, so the order is part
//! of the input. `serde_json::Value` sorts the keys of an object unless its
//! `preserve_order` feature is on, and that feature is global to a build: it
//! would change the key order of every other crate that shares it. This type
//! keeps the order locally. Numbers are `serde_json::Number`, so they read and
//! print as they do everywhere else.

use std::fmt;
use std::ops::Index;

use indexmap::IndexMap;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::Number;

/// The members of an object, in the order they were written.
pub type Object = IndexMap<String, Json>;

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Json {
    #[default]
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Json>),
    Object(Object),
}

static NULL: Json = Json::Null;

impl Json {
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }

    pub fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(truth) => Some(*truth),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Number(number) => number.as_i64(),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(number) => number.as_u64(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(number) => number.as_f64(),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Object> {
        match self {
            Self::Object(members) => Some(members),
            _ => None,
        }
    }

    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object().and_then(|members| members.get(key))
    }

    /// Sets the member `key` of an object, in place if it is there already,
    /// last otherwise. Anything but an object is left as it is.
    pub fn set(&mut self, key: &str, value: Json) {
        if let Self::Object(members) = self {
            members.insert(key.to_owned(), value);
        }
    }
}

impl PartialEq<str> for Json {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == Some(other)
    }
}

impl PartialEq<&str> for Json {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == Some(*other)
    }
}

/// `value["key"]` is `Null` when the member, or the object, is not there.
impl Index<&str> for Json {
    type Output = Json;

    fn index(&self, key: &str) -> &Json {
        self.get(key).unwrap_or(&NULL)
    }
}

/// `value[3]` is `Null` when the item, or the array, is not there.
impl Index<usize> for Json {
    type Output = Json;

    fn index(&self, index: usize) -> &Json {
        self.as_array()
            .and_then(|items| items.get(index))
            .unwrap_or(&NULL)
    }
}

impl From<&str> for Json {
    fn from(text: &str) -> Self {
        Self::String(text.to_owned())
    }
}

impl From<String> for Json {
    fn from(text: String) -> Self {
        Self::String(text)
    }
}

/// An unordered value gets its keys in alphabetical order: what its objects
/// already are.
impl From<serde_json::Value> for Json {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(truth) => Self::Bool(truth),
            serde_json::Value::Number(number) => Self::Number(number),
            serde_json::Value::String(text) => Self::String(text),
            serde_json::Value::Array(items) => {
                Self::Array(items.into_iter().map(Self::from).collect())
            }
            serde_json::Value::Object(members) => Self::Object(
                members
                    .into_iter()
                    .map(|(key, value)| (key, Self::from(value)))
                    .collect(),
            ),
        }
    }
}

/// Compact JSON.
impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        f.write_str(&text)
    }
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(truth) => serializer.serialize_bool(*truth),
            Self::Number(number) => number.serialize(serializer),
            Self::String(text) => serializer.serialize_str(text),
            Self::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Self::Object(members) => {
                let mut map = serializer.serialize_map(Some(members.len()))?;
                for (key, value) in members {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JsonVisitor;

        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = Json;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("any JSON value")
            }

            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }

            fn visit_none<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }

            fn visit_bool<E>(self, truth: bool) -> Result<Json, E> {
                Ok(Json::Bool(truth))
            }

            fn visit_i64<E>(self, number: i64) -> Result<Json, E> {
                Ok(Json::Number(number.into()))
            }

            fn visit_u64<E>(self, number: u64) -> Result<Json, E> {
                Ok(Json::Number(number.into()))
            }

            fn visit_f64<E: de::Error>(self, number: f64) -> Result<Json, E> {
                Number::from_f64(number)
                    .map(Json::Number)
                    .ok_or_else(|| E::custom("a JSON number must be finite"))
            }

            fn visit_str<E>(self, text: &str) -> Result<Json, E> {
                Ok(Json::String(text.to_owned()))
            }

            fn visit_string<E>(self, text: String) -> Result<Json, E> {
                Ok(Json::String(text))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(1024));
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Json::Array(items))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut members = Object::new();
                while let Some((key, value)) = map.next_entry::<String, Json>()? {
                    members.insert(key, value);
                }
                Ok(Json::Object(members))
            }
        }

        deserializer.deserialize_any(JsonVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Json {
        serde_json::from_str(text).expect("json")
    }

    #[test]
    fn objects_keep_the_order_of_their_keys_at_every_depth() {
        let text = r#"{"zeta":1,"alpha":{"y":[{"b":1,"a":2}],"x":null},"mid":"m"}"#;
        let value = parse(text);
        assert_eq!(serde_json::to_string(&value).expect("json"), text);
        assert_eq!(value.to_string(), text);
    }

    #[test]
    fn scalars_round_trip_as_serde_json_prints_them() {
        let text = r#"[1,-2,3.5,1e30,true,false,null,"é\n\"",18446744073709551615]"#;
        let value = parse(text);
        let plain: serde_json::Value = serde_json::from_str(text).expect("json");
        assert_eq!(
            serde_json::to_string(&value).expect("json"),
            serde_json::to_string(&plain).expect("json")
        );
    }

    #[test]
    fn indexing_a_missing_member_gives_null() {
        let value = parse(r#"{"a":[1,2]}"#);
        assert_eq!(value["a"][1].as_u64(), Some(2));
        assert!(value["b"].is_null());
        assert!(value["a"][2].is_null());
        assert!(value["a"]["x"].is_null());
        assert!(Json::from("s")["x"].is_null());
    }

    #[test]
    fn unordered_values_convert_in_their_own_order() {
        let value = Json::from(serde_json::json!({"b": [1], "a": "x"}));
        assert_eq!(value.to_string(), r#"{"a":"x","b":[1]}"#);
    }

    #[test]
    fn malformed_text_is_an_error() {
        assert!(serde_json::from_str::<Json>("{\"a\":").is_err());
    }
}
