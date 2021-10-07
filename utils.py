import json
from typing import Any, Callable, Dict

import requests
import tensorflow as tf


def get_activation(name: str) -> Callable:
    if name == "gelu":
        return lambda x: tf.keras.activations.gelu(x)

    raise ValueError(f"{name} not found")


def get_config(name: str) -> Dict[str, Any]:
    url = f"https://huggingface.co/{name}/resolve/main/config.json"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Cannot get config, status code: {response.status_code}, url: {url}")

    return json.loads(response.text)


def get_tokenizer_config(name: str, filename: str = "tokenizer_config.json"):
    url = f"https://huggingface.co/{name}/resolve/main/{filename}"
    response = requests.get(url)
    if response.status_code != 200:
        return {}

    return json.loads(response.text)


class BertPackInputsSavedModelWrapper(tf.train.Checkpoint):
    def __init__(self, bert_pack_inputs, bert_pack_inputs_fn):
        super().__init__()

        # Preserve the layer's configured seq_length as a default but make it
        # overridable. Having this dynamically determined default argument
        # requires self.__call__ to be defined in this indirect way.
        default_seq_length = bert_pack_inputs.seq_length

        @tf.function(autograph=False)
        def call(inputs, seq_length=default_seq_length):
            return bert_pack_inputs_fn(
                inputs,
                seq_length=seq_length,
                start_of_sequence_id=bert_pack_inputs.start_of_sequence_id,
                end_of_segment_id=bert_pack_inputs.end_of_segment_id,
                padding_id=bert_pack_inputs.padding_id,
            )

        self.__call__ = call

        for ragged_rank in range(1, 3):
            for num_segments in range(1, 3):
                _ = self.__call__.get_concrete_function(
                    [tf.RaggedTensorSpec([None] * (ragged_rank + 1), dtype=tf.int32) for _ in range(num_segments)], seq_length=tf.TensorSpec([], tf.int32)
                )
