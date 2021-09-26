# huggingface-to-tfhub

This repository contains scripts to convert models in huggingface model hub to TF SavedModel format.

## Converted Models

### DistilBERT models

```sh
./export_all_distilbert.sh
```

I patched `BertEncoder` and `create_preprocessing` in [tensorflow/models](https://github.com/tensorflow/models)(`==2.6.0`) for DistilBERT. (specifically, I removed token type embedding layer and pooler layer) Check `./distilbert.py` for details.
