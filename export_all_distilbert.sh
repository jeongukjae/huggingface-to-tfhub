#!/bin/sh

set -ex

python export_distilbert.py --model_name distilbert-base-uncased --output_dir ./models --tokenizer_type bert
python export_distilbert.py --model_name distilbert-base-cased --output_dir ./models --tokenizer_type bert
python export_distilbert.py --model_name distilbert-base-multilingual-cased --output_dir ./models --tokenizer_type bert
