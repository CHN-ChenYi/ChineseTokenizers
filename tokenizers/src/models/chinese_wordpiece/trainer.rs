use super::ChineseWordPiece;
use crate::models::bpe::{BpeTrainer, BpeTrainerBuilder, BPE};
use crate::tokenizer::{AddedToken, Result, Trainer};
use std::collections::HashSet;

/// A `ChineseWordPieceTrainerBuilder` can be used to create a `ChineseWordPieceTrainer` with a custom
/// configuration.
pub struct ChineseWordPieceTrainerBuilder {
    bpe_trainer_builder: BpeTrainerBuilder,
}

impl Default for ChineseWordPieceTrainerBuilder {
    fn default() -> Self {
        Self {
            bpe_trainer_builder: BpeTrainerBuilder::new().continuing_subword_prefix("##".into()),
        }
    }
}

impl ChineseWordPieceTrainerBuilder {
    /// Constructs a new `ChineseWordPieceTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    #[must_use]
    pub fn min_frequency(mut self, frequency: u32) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.min_frequency(frequency);
        self
    }

    /// Set the vocabulary size
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.vocab_size(size);
        self
    }

    /// Set whether to show progress
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.show_progress(show);
        self
    }

    /// Set the special tokens
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.special_tokens(tokens);
        self
    }

    /// Set whether to limit the alphabet
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.limit_alphabet(limit);
        self
    }

    /// Set the initial alphabet
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.initial_alphabet(alphabet);
        self
    }

    /// Set the continuing_subword_prefix
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.continuing_subword_prefix(prefix);
        self
    }

    /// Set the end_of_word_suffix
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.end_of_word_suffix(suffix);
        self
    }

    /// Constructs the final BpeTrainer
    pub fn build(self) -> ChineseWordPieceTrainer {
        let bpe_trainer = self.bpe_trainer_builder.build();
        ChineseWordPieceTrainer { bpe_trainer }
    }
}

/// Trains a `ChineseWordPiece` model.
#[derive(Default)]
pub struct ChineseWordPieceTrainer {
    bpe_trainer: BpeTrainer,
}

impl ChineseWordPieceTrainer {
    pub fn min_frequency(&self) -> u32 {
        self.bpe_trainer.min_frequency
    }

    pub fn set_min_frequency(&mut self, freq: u32) {
        self.bpe_trainer.min_frequency = freq;
    }

    pub fn vocab_size(&self) -> usize {
        self.bpe_trainer.vocab_size
    }

    pub fn set_vocab_size(&mut self, size: usize) {
        self.bpe_trainer.vocab_size = size;
    }

    pub fn show_progress(&self) -> bool {
        self.bpe_trainer.show_progress
    }

    pub fn set_show_progress(&mut self, show_progress: bool) {
        self.bpe_trainer.show_progress = show_progress;
    }

    pub fn special_tokens(&self) -> &[AddedToken] {
        &self.bpe_trainer.special_tokens
    }

    pub fn set_special_tokens(&mut self, special_tokens: Vec<AddedToken>) {
        self.bpe_trainer.special_tokens = special_tokens;
    }

    pub fn limit_alphabet(&self) -> Option<usize> {
        self.bpe_trainer.limit_alphabet
    }

    pub fn set_limit_alphabet(&mut self, limit: Option<usize>) {
        self.bpe_trainer.limit_alphabet = limit;
    }

    pub fn initial_alphabet(&self) -> &HashSet<char> {
        &self.bpe_trainer.initial_alphabet
    }

    pub fn set_initial_alphabet(&mut self, alphabet: HashSet<char>) {
        self.bpe_trainer.initial_alphabet = alphabet;
    }

    pub fn continuing_subword_prefix(&self) -> &Option<String> {
        &self.bpe_trainer.continuing_subword_prefix
    }

    pub fn set_continuing_subword_prefix(&mut self, prefix: Option<String>) {
        self.bpe_trainer.continuing_subword_prefix = prefix;
    }

    pub fn end_of_word_suffix(&self) -> &Option<String> {
        &self.bpe_trainer.end_of_word_suffix
    }

    pub fn set_end_of_word_suffix(&mut self, suffix: Option<String>) {
        self.bpe_trainer.end_of_word_suffix = suffix;
    }

    pub fn builder() -> ChineseWordPieceTrainerBuilder {
        ChineseWordPieceTrainerBuilder::default()
    }

    pub fn train(&self, model: &mut ChineseWordPiece) -> Result<Vec<AddedToken>> {
        let mut bpe = BPE::default();
        let special_tokens = self.bpe_trainer.train(&mut bpe)?;
        let new_ChineseWordPiece = ChineseWordPiece::from_bpe(&bpe);

        // Transfer the vocab
        model.vocab = new_ChineseWordPiece.vocab;
        model.vocab_r = new_ChineseWordPiece.vocab_r;
        // The continuing_subword_prefix is the only other option to be overriden by the trainer
        model.continuing_subword_prefix = new_ChineseWordPiece.continuing_subword_prefix;

        Ok(special_tokens)
    }
}

impl Trainer for ChineseWordPieceTrainer {
    type Model = ChineseWordPiece;

    fn train(&self, model: &mut ChineseWordPiece) -> Result<Vec<AddedToken>> {
        self.train(model)
    }

    fn should_show_progress(&self) -> bool {
        self.bpe_trainer.should_show_progress()
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        self.bpe_trainer.feed(iterator, process)
    }
}
