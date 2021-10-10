# huggingface-to-tfhub

This repository contains scripts to convert models in huggingface model hub to TF SavedModel format.

## Converted Models

### RoBERTa models

- [huggingface - `roberta-base`](https://huggingface.co/roberta-base)
  - tfhub: WIP
- [huggingface - `roberta-large`](https://huggingface.co/roberta-large)
  - tfhub: WIP
- [huggingface - `xlm-roberta-base`](https://huggingface.co/xlm-roberta-base)
  - tfhub: WIP
- [huggingface - `xlm-roberta-large`](https://huggingface.co/xlm-roberta-large)
  - tfhub: WIP

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

I patched `BertEncoder` and `create_preprocessing` in [tensorflow/models](https://github.com/tensorflow/models)(`==2.6.0`) for DistilBERT. (specifically, I removed token type embedding layer and pooler layer) Check `./distilbert.py` for details.

**How to use?**

You can use converted DistilBERT models directly via tfhub api. For example, you can use multilingual distilbert model as follows.

```python
# define a text embedding model
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/distilbert_multi_cased_preprocess/1")
encoder_inputs = preprocessor(text_input)

encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/distilbert_multi_cased_L-6_H-768_A-12/1", trainable=True)
encoder_outputs = encoder(encoder_inputs)
pooled_output = encoder_outputs["pooled_output"]      # [batch_size, 768].
sequence_output = encoder_outputs["sequence_output"]  # [batch_size, seq_length, 768].

model = tf.keras.Model(encoder_inputs, pooled_output)

# You can embed your sentences as follows
sentences = tf.constant(["(your text here)"])
print(embedding_model(sentences))
```
