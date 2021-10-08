#!/bin/sh

set -ex

python export_roberta.py \
    --model_name roberta-base --tokenizer_type bpe \
    --output_dir ./models --output_model_name roberta_en_cased

# It is okay to overwrite preprocess model (roberta-base and roberta-large have same preprocess model)
python export_roberta.py \
    --model_name roberta-large --tokenizer_type bpe \
    --output_dir ./models --output_model_name roberta_en_cased


python export_roberta.py \
    --model_name xlm-roberta-base --tokenizer_type xlm-spm \
    --output_dir ./models --output_model_name xlm_roberta_cased

# It is okay to overwrite preprocess model (xlm-roberta-base and xlm-roberta-large have same preprocess model)
python export_roberta.py \
    --model_name xlm-roberta-large --tokenizer_type xlm-spm \
    --output_dir ./models --output_model_name xlm_roberta_cased
