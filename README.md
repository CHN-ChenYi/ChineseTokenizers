# ChineseTokenizers

A tokenizer based on [Tokenizers](https://github.com/huggingface/tokenizers).

## Additional features

* Jieba Pre-tokenizer
* ChineseWordPiece Model (based on [Yuan-1.0](https://github.com/Shawn-Inspur/Yuan-1.0))

## Examples

### Yuan Preprocessor

``` bash
RAYON_NUM_THREADS=48 TOKENIZERS_PARALLELISM=1 cargo run --release --example yuan
```
