#!/bin/sh

set -ex

python export_distilbert.py \
    --model_name distilbert-base-uncased --tokenizer_type bert \
    --output_dir ./models --output_model_name distilbert_en_uncased

python export_distilbert.py \
    --model_name distilbert-base-cased --tokenizer_type bert \
    --output_dir ./models --output_model_name distilbert_en_cased

python export_distilbert.py \
    --model_name distilbert-base-multilingual-cased --tokenizer_type bert \
    --output_dir ./models --output_model_name distilbert_multi_cased
