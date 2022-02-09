use super::{super::OrderedVocabIter, ChineseWordPiece, ChineseWordPieceBuilder};
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::collections::HashSet;

impl Serialize for ChineseWordPiece {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("ChineseWordPiece", 5)?;

        // Small fields first
        model.serialize_field("type", "ChineseWordPiece")?;
        model.serialize_field("unk_token", &self.unk_token)?;
        model.serialize_field("continuing_subword_prefix", &self.continuing_subword_prefix)?;
        model.serialize_field("max_input_chars_per_word", &self.max_input_chars_per_word)?;

        // Then large ones
        let ordered_vocab = OrderedVocabIter::new(&self.vocab_r);
        model.serialize_field("vocab", &ordered_vocab)?;

        model.end()
    }
}

impl<'de> Deserialize<'de> for ChineseWordPiece {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "ChineseWordPiece",
            &[
                "type",
                "unk_token",
                "continuing_subword_prefix",
                "max_input_chars_per_word",
                "vocab",
            ],
            ChineseWordPieceVisitor,
        )
    }
}

struct ChineseWordPieceVisitor;
impl<'de> Visitor<'de> for ChineseWordPieceVisitor {
    type Value = ChineseWordPiece;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct ChineseWordPiece")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = ChineseWordPieceBuilder::new();
        let mut missing_fields = vec![
            // for retrocompatibility the "type" field is not mandatory
            "unk_token",
            "continuing_subword_prefix",
            "max_input_chars_per_word",
            "vocab",
        ]
        .into_iter()
        .collect::<HashSet<_>>();

        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "unk_token" => builder = builder.unk_token(map.next_value()?),
                "continuing_subword_prefix" => {
                    builder = builder.continuing_subword_prefix(map.next_value()?)
                }
                "max_input_chars_per_word" => {
                    builder = builder.max_input_chars_per_word(map.next_value()?)
                }
                "vocab" => builder = builder.vocab(map.next_value()?),
                "type" => match map.next_value()? {
                    "ChineseWordPiece" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"ChineseWordPiece",
                        ))
                    }
                },
                _ => {}
            }
            missing_fields.remove::<str>(&key);
        }

        if !missing_fields.is_empty() {
            Err(serde::de::Error::missing_field(
                missing_fields.iter().next().unwrap(),
            ))
        } else {
            Ok(builder.build().map_err(serde::de::Error::custom)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde() {
        let wp = ChineseWordPiece::default();
        let wp_s = "{\
            \"type\":\"ChineseWordPiece\",\
            \"unk_token\":\"[UNK]\",\
            \"continuing_subword_prefix\":\"##\",\
            \"max_input_chars_per_word\":100,\
            \"vocab\":{}\
        }";

        assert_eq!(serde_json::to_string(&wp).unwrap(), wp_s);
        assert_eq!(serde_json::from_str::<ChineseWordPiece>(wp_s).unwrap(), wp);
    }

    #[test]
    fn deserialization_should_fail() {
        let missing_unk = "{\
            \"type\":\"ChineseWordPiece\",\
            \"continuing_subword_prefix\":\"##\",\
            \"max_input_chars_per_word\":100,\
            \"vocab\":{}\
        }";
        assert!(serde_json::from_str::<ChineseWordPiece>(missing_unk)
            .unwrap_err()
            .to_string()
            .starts_with("missing field `unk_token`"));

        let wrong_type = "{\
            \"type\":\"WordLevel\",\
            \"unk_token\":\"[UNK]\",\
            \"vocab\":{}\
        }";
        assert!(serde_json::from_str::<ChineseWordPiece>(wrong_type)
            .unwrap_err()
            .to_string()
            .starts_with("invalid value: string \"WordLevel\", expected ChineseWordPiece"));
    }
}
