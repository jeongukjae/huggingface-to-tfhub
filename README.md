# huggingface-to-tfhub

This repository contains scripts to convert models in huggingface model hub to TF SavedModel format for my personal projects. (Mainly small multilingual models)

## Converted Models

### RoBERTa models

- [huggingface - `roberta-base`](https://huggingface.co/roberta-base)
  - encoder: [tfhub: `jeongukjae/roberta_en_cased_L-12_H-768_A-12/1`](https://tfhub.dev/jeongukjae/roberta_en_cased_L-12_H-768_A-12/1)
  - preprocessor: [tfhub: `jeongukjae/roberta_en_cased_preprocess/1`](https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1)
- [huggingface - `roberta-large`](https://huggingface.co/roberta-large)
  - encoder: [tfhub: `jeongukjae/roberta_en_cased_L-24_H-1024_A-16/1`](https://tfhub.dev/jeongukjae/roberta_en_cased_L-24_H-1024_A-16/1)
  - preprocessor: [tfhub: `jeongukjae/roberta_en_cased_preprocess/1`](https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1)
- [huggingface - `xlm-roberta-base`](https://huggingface.co/xlm-roberta-base)
  - encoder: [tfhub: `jeongukjae/xlm_roberta_multi_cased_L-12_H-768_A-12/1`](https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_L-12_H-768_A-12/1)
  - preprocessor: [tfhub: `jeongukjae/xlm_roberta_multi_cased_preprocess/1`](https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_preprocess/1)
- [huggingface - `xlm-roberta-large`](https://huggingface.co/xlm-roberta-large)
  - encoder: [tfhub: `jeongukjae/xlm_roberta_multi_cased_L-24_H-1024_A-16/1`](https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_L-24_H-1024_A-16/1)
  - preprocessor: [tfhub: `jeongukjae/xlm_roberta_multi_cased_preprocess/1`](https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_preprocess/1)

**How to convert above models?**

```sh
./export_roberta_all.sh
```

### DistilBERT models

- [huggingface - `distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
  - encoder: [tfhub - `jeongukjae/distilbert_en_uncased_L-6_H-768_A-12`](https://tfhub.dev/jeongukjae/distilbert_en_uncased_L-6_H-768_A-12/1)
  - preprocessor: [tfhub - `jeongukjae/distilbert_en_uncased_preprocess`](https://tfhub.dev/jeongukjae/distilbert_en_uncased_preprocess/1)
- [huggingface - `distilbert-base-cased`](https://huggingface.co/distilbert-base-cased)
  - encoder: [tfhub - `jeongukjae/distilbert_en_cased_L-6_H-768_A-12`](https://tfhub.dev/jeongukjae/distilbert_en_cased_L-6_H-768_A-12/1)
  - preprocessor: [tfhub - `jeongukjae/distilbert_en_cased_preprocess`](https://tfhub.dev/jeongukjae/distilbert_en_cased_preprocess/1)
- [huggingface - `distilbert-base-multilingual-cased`](https://huggingface.co/distilbert-base-multilingual-cased)
  - encoder: [tfhub - `jeongukjae/distilbert_multi_cased_L-6_H-768_A-12`](https://tfhub.dev/jeongukjae/distilbert_multi_cased_L-6_H-768_A-12/1)
  - preprocessor: [tfhub - `jeongukjae/distilbert_multi_cased_preprocess`](https://tfhub.dev/jeongukjae/distilbert_multi_cased_preprocess/1)

**How to convert above models?**

```sh
./export_distilbert_all.sh
```

### DistilKoBERT model

- [huggingface - `monologg/distilkobert`](https://huggingface.co/monologg/distilkobert)
  - encoder: [tfhub - `jeongukjae/distilkobert_cased_L-3_H-768_A-12`](https://tfhub.dev/jeongukjae/distilkobert_cased_L-3_H-768_A-12/1)
  - preprocessor: [tfhub - `jeongukjae/distilkobert_cased_preprocess`](https://tfhub.dev/jeongukjae/distilkobert_cased_preprocess/1)

**How to convert above model?**

```sh
./export_distilkobert.sh
```

### KLUE PLMs

- [huggingface - `klue/roberta-small`](https://huggingface.co/klue/roberta-small)
  - encoder: [tfhub - `jeongukjae/klue_roberta_cased_L-6_H-768_A-12`](https://tfhub.dev/jeongukjae/klue_roberta_cased_L-6_H-768_A-12/1)
  - preprocessor: [tfhub - `jeongukjae/klue_roberta_cased_preprocess`](https://tfhub.dev/jeongukjae/klue_roberta_cased_preprocess/1)
- [huggingface - `klue/roberta-base`](https://huggingface.co/klue/roberta-base)
  - encoder: [tfhub - `jeongukjae/klue_roberta_cased_L-12_H-768_A-12`](https://tfhub.dev/jeongukjae/klue_roberta_cased_L-12_H-768_A-12/1)
  - preprocessor: [tfhub - `jeongukjae/klue_roberta_cased_preprocess`](https://tfhub.dev/jeongukjae/klue_roberta_cased_preprocess/1)
- [huggingface - `klue/roberta-large`](https://huggingface.co/klue/roberta-large)
  - encoder: [tfhub - `jeongukjae/klue_roberta_cased_L-24_H-1024_A-16`](https://tfhub.dev/jeongukjae/klue_roberta_cased_L-24_H-1024_A-16/1)
  - preprocessor: [tfhub - `jeongukjae/klue_roberta_cased_preprocess`](https://tfhub.dev/jeongukjae/klue_roberta_cased_preprocess/1)
- [huggingface - `klue/bert-base`](https://huggingface.co/klue/bert-base)
  - encoder: [tfhub - `jeongukjae/klue_bert_cased_L-12_H-768_A-12`](https://tfhub.dev/jeongukjae/klue_bert_cased_L-12_H-768_A-12/1)
  - preprocessor: [tfhub - `jeongukjae/klue_bert_cased_preprocess`](https://tfhub.dev/jeongukjae/klue_bert_cased_preprocess/1)

**How to convert above model?**

```sh
./export_klue.sh
```
