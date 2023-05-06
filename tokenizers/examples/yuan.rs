use std::{
  fs,
  fs::File,
  io::{prelude::*, BufReader},
  path::Path,
  time::Instant,
};

use tokenizers::models::chinese_wordpiece::ChineseWordPiece;
use tokenizers::pre_tokenizers::jieba::Jieba;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use ndarray::Array2;
use ndarray_npy::NpzWriter;

fn lines_from_file(filename: impl AsRef<Path>) -> Vec<String> {
  let file = File::open(filename).expect("no such file");
  let buf = BufReader::new(file);
  buf
    .lines()
    .map(|l| l.expect("Could not parse line"))
    .filter(|l| l.len() >= 8192)
    .collect()
}

fn main() {
  let now = Instant::now();

  let vocab_path = "./examples/vocab.txt";
  //let data_dir = "/shared/YuanDataset/raw";
	let data_dir = "/mnt/asc22/YuanDataset/raw/";
  let output_path = "~/processed_data.npz";
  println!("vocab_path: {}", vocab_path);
  println!("data_dir: {}", data_dir);
  println!("output_path: {}", output_path);

  let mut data = Vec::new();
  let paths = fs::read_dir(data_dir).unwrap();
  for path in paths {
    data.append(&mut lines_from_file(path.as_ref().unwrap().path()));
    println!(
      "Read file: {}, lines in total: {}",
      path.unwrap().path().display(),
      data.len()
    );
    if data.len() > 500000 {
      break;
    }
  }
  data.resize(500000, "".to_string());
  println!("Reading files takes {} seconds", now.elapsed().as_secs());

  let mut tokenizer =
    Tokenizer::new(ChineseWordPiece::from_file(vocab_path).unk_token(String::from("<unk>")).build().unwrap()).into_inner();
  tokenizer.with_pre_tokenizer(Jieba::default());
  tokenizer.with_truncation(Some(TruncationParams {
    max_length: 2048,
    ..TruncationParams::default()
  }));
  tokenizer.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::Fixed(2048),
    pad_id: tokenizer.token_to_id("[PAD]").unwrap(),
    pad_token: "[PAD]".to_string(),
    ..PaddingParams::default()
  }));

  let output = tokenizer.encode_batch(data, true).unwrap();

  let mut ids = Vec::new();
  let mut attention_masks = Vec::new();
  for s in output.iter() {
    ids.append(&mut s.get_ids().iter().map(|t| *t as u16).collect::<Vec<u16>>());
    attention_masks.append(
      &mut s
        .get_attention_mask()
        .iter()
        .map(|t| *t as u16)
        .collect::<Vec<u16>>(),
    );
  }
  let np_ids = Array2::from_shape_vec((output.len(), 2048), ids).unwrap();
  let np_attention_masks = Array2::from_shape_vec((output.len(), 2048), attention_masks).unwrap();

  let mut npz = NpzWriter::new(File::create(output_path).unwrap());
  npz.add_array("id", &np_ids).expect("failed to pack id");
  npz
    .add_array("attention_mask", &np_attention_masks)
    .expect("failed to pack attention_mask");
  npz.finish().expect("failed to write npz");

  println!("Takes {} seconds in total", now.elapsed().as_secs());
  println!("Token number: {}", output.len() * 2048);
}
