# huggingface-to-tfhub

This repository contains scripts to convert models in huggingface model hub to TF SavedModel format.

## Converted Models

### DistilBERT models

| huggingface model hub name           | converted encoder name                  | converted preprocess name           |
| ------------------------------------ | --------------------------------------- | ----------------------------------- |
| `distilbert-base-uncased`            | `distilbert_en_uncased_L-6_H-768_A-12`  | `distilbert_en_uncased_preprocess`  |
| `distilbert-base-cased`              | `distilbert_en_cased_L-6_H-768_A-12`    | `distilbert_en_cased_preprocess`    |
| `distilbert-base-multilingual-cased` | `distilbert_multi_cased_L-6_H-768_A-12` | `distilbert_multi_cased_preprocess` |

**How to convert above models?**

```sh
./export_all_distilbert.sh
```

I patched `BertEncoder` and `create_preprocessing` in [tensorflow/models](https://github.com/tensorflow/models)(`==2.6.0`) for DistilBERT. (specifically, I removed token type embedding layer and pooler layer) Check `./distilbert.py` for details.
