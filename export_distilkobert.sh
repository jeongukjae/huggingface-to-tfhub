#!/bin/sh

set -ex

# https://github.com/SKTBrain/KoBERT/blob/8df69ec6b588ae661bef98d28ec29448482bbe6e/kobert/utils.py#L30
python export_distilbert.py \
    --model_name "monologg/distilkobert" --tokenizer_type spm \
    --output_dir ./models --output_model_name distilbert_ko_cased \
    --spm_path https://kobert.blob.core.windows.net/models/kobert/tokenizer/kobert_news_wiki_ko_cased-ae5711deb3.spiece \
    --padding_token "[PAD]"
