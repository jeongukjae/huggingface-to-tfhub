import enum
import json
import os
from typing import Any, Callable, Dict
from urllib import request as urllib_request

import numpy as np
import requests
import tensorflow as tf
import torch
from absl import logging
from transformers import DistilBertModel

from distilbert import DistilBertEncoder, create_distilbert_preprocessing


class TokenizerType(enum.Enum):
    BERT = "bert"
    SENTENCEPIECE = "spm"


def convert_distilbert(
    model_name: str,
    output_dir: str,
    tokenizer_type: TokenizerType = TokenizerType.BERT,
    temp_dir: str = "./tmp",
):
    os.makedirs(temp_dir, exist_ok=True)

    logging.info("Get config")
    config = get_config(model_name)
    logging.info(f"Config: {config}")

    logging.info("Build model")
    inputs = {
        "input_word_ids": tf.keras.Input([None], dtype=tf.int32, name="input_word_ids"),
        "input_mask": tf.keras.Input([None], dtype=tf.int32, name="input_mask"),
    }
    encoder = DistilBertEncoder(
        vocab_size=config["vocab_size"],
        hidden_size=config["dim"],
        num_layers=config["n_layers"],
        num_attention_heads=config["n_heads"],
        max_sequence_length=config["max_position_embeddings"],
        inner_dim=config["hidden_dim"],
        inner_activation=get_activation(config["activation"]),
        output_dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        initializer=tf.keras.initializers.TruncatedNormal(stddev=config["initializer_range"]),
    )
    model = tf.keras.Model(inputs, encoder(inputs))
    model.summary()

    logging.info("Get weights")
    torch_model = DistilBertModel.from_pretrained(model_name)
    state_dict = torch_model.state_dict()

    logging.info("Convert weights")
    encoder._embedding_layer.set_weights(
        [
            state_dict["embeddings.word_embeddings.weight"],
        ]
    )
    encoder._position_embedding_layer.set_weights([state_dict["embeddings.position_embeddings.weight"]])
    encoder._embedding_norm_layer.set_weights(
        [
            state_dict["embeddings.LayerNorm.weight"],
            state_dict["embeddings.LayerNorm.bias"],
        ]
    )

    for layer_index in range(config["n_layers"]):
        encoder._transformer_layers[layer_index]._attention_layer._query_dense.set_weights(
            [
                tf.reshape(state_dict[f"transformer.layer.{layer_index}.attention.q_lin.weight"].T, [config["dim"], config["n_heads"], -1]),
                tf.reshape(state_dict[f"transformer.layer.{layer_index}.attention.q_lin.bias"], [config["n_heads"], -1]),
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer._key_dense.set_weights(
            [
                tf.reshape(state_dict[f"transformer.layer.{layer_index}.attention.k_lin.weight"].T, [config["dim"], config["n_heads"], -1]),
                tf.reshape(state_dict[f"transformer.layer.{layer_index}.attention.k_lin.bias"], [config["n_heads"], -1]),
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer._value_dense.set_weights(
            [
                tf.reshape(state_dict[f"transformer.layer.{layer_index}.attention.v_lin.weight"].T, [config["dim"], config["n_heads"], -1]),
                tf.reshape(state_dict[f"transformer.layer.{layer_index}.attention.v_lin.bias"], [config["n_heads"], -1]),
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer._output_dense.set_weights(
            [
                tf.reshape(
                    state_dict[f"transformer.layer.{layer_index}.attention.out_lin.weight"].T, [config["n_heads"], -1, config["dim"]]
                ),
                state_dict[f"transformer.layer.{layer_index}.attention.out_lin.bias"],
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer_norm.set_weights(
            [
                state_dict[f"transformer.layer.{layer_index}.sa_layer_norm.weight"],
                state_dict[f"transformer.layer.{layer_index}.sa_layer_norm.bias"],
            ]
        )
        encoder._transformer_layers[layer_index]._intermediate_dense.set_weights(
            [
                state_dict[f"transformer.layer.{layer_index}.ffn.lin1.weight"].T,
                state_dict[f"transformer.layer.{layer_index}.ffn.lin1.bias"],
            ]
        )
        encoder._transformer_layers[layer_index]._output_dense.set_weights(
            [
                state_dict[f"transformer.layer.{layer_index}.ffn.lin2.weight"].T,
                state_dict[f"transformer.layer.{layer_index}.ffn.lin2.bias"],
            ]
        )
        encoder._transformer_layers[layer_index]._output_layer_norm.set_weights(
            [
                state_dict[f"transformer.layer.{layer_index}.output_layer_norm.weight"],
                state_dict[f"transformer.layer.{layer_index}.output_layer_norm.bias"],
            ]
        )

    # assertion
    logging.info("Forward models and check outptus")
    input_word_ids = tf.random.uniform(shape=[32, 512], minval=1, maxval=config["vocab_size"], dtype=tf.int32).numpy()
    input_mask = tf.random.uniform(shape=[32, 512], minval=0, maxval=2, dtype=tf.int32).numpy()

    torch_output = (
        torch_model(
            input_ids=torch.tensor(input_word_ids),
            attention_mask=torch.tensor(input_mask),
        )
        .last_hidden_state.detach()
        .numpy()
    )
    tf_output = model(
        {
            "input_word_ids": input_word_ids,
            "input_mask": input_mask,
        }
    )["sequence_output"].numpy()
    np.testing.assert_allclose(tf_output, torch_output, rtol=1e-5, atol=1e-5)

    logging.info("Save model")
    model.save(os.path.join(output_dir, model_name))

    if tokenizer_type == TokenizerType.BERT:
        vocab_file = os.path.join(temp_dir, "vocab.txt")
        urllib_request.urlretrieve(f"https://huggingface.co/{model_name}/resolve/main/vocab.txt", vocab_file)
        tokenizer_config = get_tokenizer_config(model_name)
        do_lower_case = tokenizer_config["do_lower_case"] if "do_lower_case" in tokenizer_config else False

        logging.info(f"do_lower_case: {do_lower_case}")
        preprocessor = create_distilbert_preprocessing(vocab_file=vocab_file, do_lower_case=do_lower_case)
        preprocessor.save(os.path.join(output_dir, model_name + "_preprocess"))


def get_config(name: str) -> Dict[str, Any]:
    url = f"https://huggingface.co/{name}/resolve/main/config.json"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Cannot get config, status code: {response.status_code}, url: {url}")

    return json.loads(response.text)


def get_tokenizer_config(name: str):
    url = f"https://huggingface.co/{name}/resolve/main/tokenizer_config.json"
    response = requests.get(url)
    if response.status_code != 200:
        return {}

    return json.loads(response.text)


def get_activation(name: str) -> Callable:
    if name == "gelu":
        return lambda x: tf.keras.activations.gelu(x)

    raise ValueError(f"{name} not found")
