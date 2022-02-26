//! [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
//! model.

use crate::models::bpe::BPE;
use crate::tokenizer::{Model, Result, Token};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

mod serialization;
mod trainer;
pub use trainer::*;

fn contain_chinese(s: &str) -> bool {
    for c in s.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&c) {
            return true;
        }
    }
    false
}

#[derive(Debug)]
pub enum Error {
    MissingUnkToken,
}
impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::MissingUnkToken => write!(
                fmt,
                "ChineseWordPiece error: Missing [UNK] token from the vocabulary"
            ),
        }
    }
}

type Vocab = HashMap<String, u32>;
type VocabR = HashMap<u32, String>;

struct Config {
    files: Option<String>,
    vocab: Vocab,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

/// A `ChineseWordPieceBuilder` can be used to create a `ChineseWordPiece` model with a custom configuration.
pub struct ChineseWordPieceBuilder {
    config: Config,
}

impl Default for ChineseWordPieceBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                files: None,
                vocab: HashMap::new(),
                unk_token: String::from("[UNK]"),
                continuing_subword_prefix: String::from("##"),
                max_input_chars_per_word: 100,
            },
        }
    }
}

impl ChineseWordPieceBuilder {
    /// Construct a new `ChineseWordPieceBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input files.
    #[must_use]
    pub fn files(mut self, vocab: String) -> Self {
        self.config.files = Some(vocab);
        self
    }

    /// Set the vocab (token -> ID) mapping.
    #[must_use]
    pub fn vocab(mut self, vocab: Vocab) -> Self {
        self.config.vocab = vocab;
        self
    }

    /// The the `UNK` token for the vocab.
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = unk_token;
        self
    }

    /// Set the prefix for continuing subwords.
    #[must_use]
    pub fn continuing_subword_prefix(mut self, continuing_subword_prefix: String) -> Self {
        self.config.continuing_subword_prefix = continuing_subword_prefix;
        self
    }

    /// Set the maximum number of input characters per word.
    #[must_use]
    pub fn max_input_chars_per_word(mut self, max_input_chars_per_word: usize) -> Self {
        self.config.max_input_chars_per_word = max_input_chars_per_word;
        self
    }

    /// Contructs a `ChineseWordPiece` model that uses the `ChineseWordPieceBuilder`'s configuration.
    pub fn build(mut self) -> Result<ChineseWordPiece> {
        if let Some(ref vocab) = self.config.files {
            self.config.vocab = ChineseWordPiece::read_file(&vocab)?;
        }

        let trie = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(
                self.config
                    .vocab
                    .iter()
                    .map(|(key, _)| {
                        if contain_chinese(key) {
                            return key.clone();
                        }
                        if key.starts_with(&self.config.continuing_subword_prefix) {
                            let mut new_key = key.clone();
                            new_key
                                .replace_range(..self.config.continuing_subword_prefix.len(), "");
                            return new_key;
                        }
                        String::from("")
                    })
                    .filter(|key| key.len() > 0)
                    .collect::<Vec<String>>(),
            );

        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();

        Ok(ChineseWordPiece {
            vocab: self.config.vocab,
            vocab_r,
            trie,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            max_input_chars_per_word: self.config.max_input_chars_per_word,
        })
    }
}

/// A
/// [ChineseWordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
/// model.
#[derive(Clone)]
pub struct ChineseWordPiece {
    vocab: Vocab,
    vocab_r: VocabR,
    trie: AhoCorasick,
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
}

impl PartialEq for ChineseWordPiece {
    fn eq(&self, other: &Self) -> bool {
        self.vocab == other.vocab
            && self.unk_token == other.unk_token
            && self.continuing_subword_prefix == other.continuing_subword_prefix
            && self.max_input_chars_per_word == other.max_input_chars_per_word
    }
}

impl std::fmt::Debug for ChineseWordPiece {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("ChineseWordPiece")
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("max_input_chars_per_word", &self.max_input_chars_per_word)
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

impl Default for ChineseWordPiece {
    fn default() -> Self {
        Self {
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            trie: AhoCorasick::new(&[""]),
            unk_token: String::from("[UNK]"),
            continuing_subword_prefix: String::from("##"),
            max_input_chars_per_word: 100,
        }
    }
}

impl ChineseWordPiece {
    /// Get a `ChineseWordPieceBuilder`.
    pub fn builder() -> ChineseWordPieceBuilder {
        ChineseWordPieceBuilder::new()
    }

    /// Read the given files to extract the vocab
    pub fn read_file(vocab: &str) -> Result<Vocab> {
        let file = File::open(vocab)?;
        let file = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab.insert(line.trim_end().to_owned(), index as u32);
        }

        Ok(vocab)
    }

    /// Initialize a `ChineseWordPiece` model from a vocab mapping file.
    pub fn from_file(vocab: &str) -> ChineseWordPieceBuilder {
        ChineseWordPiece::builder().files(vocab.to_owned())
    }

    /// Create a `ChineseWordPiece` model from a `BPE` model.
    pub fn from_bpe(bpe: &BPE) -> Self {
        let mut wp = Self::builder().vocab(bpe.get_vocab()).build().unwrap();
        if let Some(unk) = bpe.get_unk_token() {
            wp.unk_token = unk.to_owned();
        }
        if let Some(prefix) = bpe.get_continuing_subword_prefix() {
            wp.continuing_subword_prefix = prefix.to_owned();
        }
        wp
    }
}

impl Model for ChineseWordPiece {
    type Trainer = ChineseWordPieceTrainer;

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        let char_len = sequence.chars().count();
        if char_len > self.max_input_chars_per_word {
            return Ok(vec![Token {
                value: self.unk_token.clone(),
                id: *self
                    .vocab
                    .get(&self.unk_token)
                    .ok_or(Error::MissingUnkToken)?,
                offsets: (0, sequence.len()),
            }]);
        }

        let mut sub_tokens: Vec<Token> = vec![];

        let mut chars_index: Vec<usize> = vec![];
        let mut chars_index_ = 0usize;

        chars_index.push(0);
        for c in sequence.chars() {
            chars_index_ += c.len_utf8();
            chars_index.push(chars_index_);
        }

        // get first token
        let offset_start = 0;
        let mut end = chars_index.len() - 1;
        while end > 0 {
            let offset_end = chars_index[end];
            let substr: Cow<str> = Cow::Borrowed(&sequence[offset_start..offset_end]);
            if self.vocab.contains_key(substr.as_ref()) {
                sub_tokens.push(Token {
                    id: self.vocab[substr.as_ref()],
                    value: substr.to_string(),
                    offsets: (offset_start, offset_end),
                });
                break;
            }
            end -= 1;
        }
        if end == 0 {
            sub_tokens.push(Token {
                value: self.unk_token.clone(),
                id: *self
                    .vocab
                    .get(&self.unk_token)
                    .ok_or(Error::MissingUnkToken)?,
                offsets: (0, chars_index[1]),
            });
            end += 1;
        }

        // get other tokens
        let mut last_id = end;
        if last_id < chars_index.len() {
            let offset = chars_index[last_id];
            for mat in self.trie.find_iter(&sequence[offset..]) {
                let mut last_offset = chars_index[last_id];
                let offset_start = mat.start() + offset;
                let offset_end = mat.end() + offset;
                if offset_start != last_offset {
                    while offset_start != last_offset {
                        sub_tokens.push(Token {
                            value: self.unk_token.clone(),
                            id: *self
                                .vocab
                                .get(&self.unk_token)
                                .ok_or(Error::MissingUnkToken)?,
                            offsets: (last_offset, chars_index[last_id + 1]),
                        });
                        last_id += 1;
                        last_offset = chars_index[last_id];
                    }
                }
                let mut substr: Cow<str> = Cow::Borrowed(&sequence[offset_start..offset_end]);
                if !contain_chinese(substr.as_ref()) {
                    substr = Cow::Owned(format!("{}{}", self.continuing_subword_prefix, substr));
                }
                sub_tokens.push(Token {
                    id: self.vocab[substr.as_ref()],
                    value: substr.to_string(),
                    offsets: (offset_start, offset_end),
                });
                while chars_index[last_id] != offset_end {
                    last_id += 1;
                }
            }
        }

        // get remaining unk
        last_id += 1;
        while last_id < chars_index.len() {
            sub_tokens.push(Token {
                value: self.unk_token.clone(),
                id: *self
                    .vocab
                    .get(&self.unk_token)
                    .ok_or(Error::MissingUnkToken)?,
                offsets: (chars_index[last_id - 1], chars_index[last_id]),
            });
            last_id += 1;
        }

        Ok(sub_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_file_name = match name {
            Some(name) => format!("{}-vocab.txt", name),
            None => "vocab.txt".to_string(),
        };

        // Write vocab.txt
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let mut vocab: Vec<(&String, &u32)> = self.vocab.iter().collect();
        vocab.sort_unstable_by_key(|k| *k.1);
        vocab_file.write_all(
            &vocab
                .into_iter()
                .flat_map(|(token, _)| format!("{}\n", token).as_bytes().to_owned())
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path])
    }

    fn get_trainer(&self) -> Self::Trainer {
        ChineseWordPieceTrainer::builder().build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert!(format!("{}", Error::MissingUnkToken).contains("Missing [UNK] token"));
    }
}
