import enum
import os
from urllib import request as urllib_request

import numpy as np
import tensorflow as tf
import tensorflow_text as text  # noqa
import torch
from absl import app, flags, logging
from official.nlp import keras_nlp
from official.nlp.modeling import layers
from official.nlp.modeling.layers import text_layers
from official.nlp.tools import export_tfhub_lib
from transformers import BertModel

from utils import get_activation, get_config, get_tokenizer_config

FLAGS = flags.FLAGS


def main(argv):
    convert_bert(
        model_name=FLAGS.model_name,
        output_dir=FLAGS.output_dir,
        output_model_name=FLAGS.output_model_name,
        tokenizer_type=TokenizerType(FLAGS.tokenizer_type),
    )


class TokenizerType(enum.Enum):
    BERT = "bert"


def convert_bert(
    model_name: str,
    output_dir: str,
    output_model_name: str,
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
        "input_type_ids": tf.keras.Input([None], dtype=tf.int32, name="input_type_ids"),
    }
    # Layer normalization's epsilon value in tf-models' BertEncoder is not parameterized,
    # so we just check the config is same.
    assert config["layer_norm_eps"] == 1e-12
    encoder = keras_nlp.encoders.BertEncoder(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_hidden_layers"],
        type_vocab_size=config["type_vocab_size"],
        num_attention_heads=config["num_attention_heads"],
        max_sequence_length=config["max_position_embeddings"],
        inner_dim=config["intermediate_size"],
        inner_activation=get_activation(config["hidden_act"]),
        output_dropout=config["hidden_dropout_prob"],
        attention_dropout=config["attention_probs_dropout_prob"],
        initializer=tf.keras.initializers.TruncatedNormal(stddev=config["initializer_range"]),
    )
    logging.info(f"Model Config: {encoder.get_config()}")
    model = tf.keras.Model(inputs, encoder(inputs))
    model.summary()

    logging.info("Get weights")
    torch_model = BertModel.from_pretrained(model_name)
    state_dict = torch_model.state_dict()

    logging.info("Convert weights")
    encoder._embedding_layer.set_weights([state_dict["embeddings.word_embeddings.weight"]])
    encoder._position_embedding_layer.set_weights([state_dict["embeddings.position_embeddings.weight"]])
    encoder._type_embedding_layer.set_weights([state_dict["embeddings.token_type_embeddings.weight"]])
    encoder._embedding_norm_layer.set_weights([state_dict["embeddings.LayerNorm.weight"], state_dict["embeddings.LayerNorm.bias"]])

    for layer_index in range(config["num_hidden_layers"]):
        encoder._transformer_layers[layer_index]._attention_layer._query_dense.set_weights(
            [
                tf.reshape(
                    state_dict[f"encoder.layer.{layer_index}.attention.self.query.weight"].T,
                    [config["hidden_size"], config["num_attention_heads"], -1],
                ),
                tf.reshape(
                    state_dict[f"encoder.layer.{layer_index}.attention.self.query.bias"],
                    [config["num_attention_heads"], -1],
                ),
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer._key_dense.set_weights(
            [
                tf.reshape(
                    state_dict[f"encoder.layer.{layer_index}.attention.self.key.weight"].T,
                    [config["hidden_size"], config["num_attention_heads"], -1],
                ),
                tf.reshape(
                    state_dict[f"encoder.layer.{layer_index}.attention.self.key.bias"],
                    [config["num_attention_heads"], -1],
                ),
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer._value_dense.set_weights(
            [
                tf.reshape(
                    state_dict[f"encoder.layer.{layer_index}.attention.self.value.weight"].T,
                    [config["hidden_size"], config["num_attention_heads"], -1],
                ),
                tf.reshape(
                    state_dict[f"encoder.layer.{layer_index}.attention.self.value.bias"],
                    [config["num_attention_heads"], -1],
                ),
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer._output_dense.set_weights(
            [
                tf.reshape(
                    state_dict[f"encoder.layer.{layer_index}.attention.output.dense.weight"].T,
                    [config["num_attention_heads"], -1, config["hidden_size"]],
                ),
                state_dict[f"encoder.layer.{layer_index}.attention.output.dense.bias"],
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer_norm.set_weights(
            [
                state_dict[f"encoder.layer.{layer_index}.attention.output.LayerNorm.weight"],
                state_dict[f"encoder.layer.{layer_index}.attention.output.LayerNorm.bias"],
            ]
        )
        encoder._transformer_layers[layer_index]._intermediate_dense.set_weights(
            [
                state_dict[f"encoder.layer.{layer_index}.intermediate.dense.weight"].T,
                state_dict[f"encoder.layer.{layer_index}.intermediate.dense.bias"],
            ]
        )
        encoder._transformer_layers[layer_index]._output_dense.set_weights(
            [
                state_dict[f"encoder.layer.{layer_index}.output.dense.weight"].T,
                state_dict[f"encoder.layer.{layer_index}.output.dense.bias"],
            ]
        )
        encoder._transformer_layers[layer_index]._output_layer_norm.set_weights(
            [
                state_dict[f"encoder.layer.{layer_index}.output.LayerNorm.weight"],
                state_dict[f"encoder.layer.{layer_index}.output.LayerNorm.bias"],
            ]
        )
    encoder._pooler_layer.set_weights(
        [
            state_dict["pooler.dense.weight"].T,
            state_dict["pooler.dense.bias"],
        ]
    )

    # assertion
    logging.info("Forward models and check outptus")
    input_word_ids = tf.random.uniform(shape=[32, 128], minval=4, maxval=config["vocab_size"], dtype=tf.int32).numpy()
    input_mask = tf.random.uniform(shape=[32, 128], minval=0, maxval=2, dtype=tf.int32).numpy()
    input_type_ids = tf.random.uniform(shape=[32, 128], minval=0, maxval=config["type_vocab_size"], dtype=tf.int32).numpy()

    with torch.no_grad():
        torch_output = torch_model(
            input_ids=torch.tensor(input_word_ids),
            attention_mask=torch.tensor(input_mask),
            token_type_ids=torch.tensor(input_type_ids),
        )
    tf_output = model(
        {
            "input_word_ids": input_word_ids,
            "input_mask": input_mask,
            "input_type_ids": input_type_ids,
        }
    )
    np.testing.assert_allclose(tf_output["sequence_output"].numpy(), torch_output.last_hidden_state.detach().numpy(), rtol=1e-3, atol=2e-4)
    np.testing.assert_allclose(tf_output["pooled_output"].numpy(), torch_output.pooler_output.detach().numpy(), rtol=1e-3, atol=2e-4)

    logging.info("Save model")
    model.save(
        os.path.join(
            output_dir,
            output_model_name + f"_L-{config['num_hidden_layers']}_H-{config['hidden_size']}_A-{config['num_attention_heads']}",
        )
    )
    if tokenizer_type == TokenizerType.BERT:
        vocab_file = os.path.join(temp_dir, "vocab.txt")
        urllib_request.urlretrieve(f"https://huggingface.co/{model_name}/resolve/main/vocab.txt", vocab_file)
        tokenizer_config = get_tokenizer_config(model_name)
        do_lower_case = tokenizer_config["do_lower_case"] if "do_lower_case" in tokenizer_config else False

        logging.info(f"do_lower_case: {do_lower_case}")
        preprocessor = create_bert_preprocessing(tokenizer_type, vocab_file=vocab_file, do_lower_case=do_lower_case)
        preprocessor.save(os.path.join(output_dir, output_model_name + "_preprocess"))
    else:
        raise ValueError


def create_bert_preprocessing(
    tokenizer_type: TokenizerType,
    *,
    vocab_file: str = None,
    do_lower_case: bool = False,
    tokenize_with_offsets: bool = True,
    default_seq_length: int = 128,
) -> tf.keras.Model:
    if tokenizer_type == TokenizerType.BERT:
        assert vocab_file
        tokenize = layers.BertTokenizer(
            vocab_file=vocab_file,
            lower_case=do_lower_case,
            tokenize_with_offsets=tokenize_with_offsets,
        )
    else:
        raise ValueError

    # The root object of the preprocessing model can be called to do
    # one-shot preprocessing for users with single-sentence inputs.
    sentences = tf.keras.layers.Input(shape=(), dtype=tf.string, name="sentences")
    if tokenize_with_offsets:
        tokens, start_offsets, limit_offsets = tokenize(sentences)
    else:
        tokens = tokenize(sentences)

    logging.info(f"Special tokens: {tokenize.get_special_tokens_dict()}")
    pack = text_layers.BertPackInputs(
        seq_length=default_seq_length,
        special_tokens_dict=tokenize.get_special_tokens_dict(),
    )
    model_inputs = pack(tokens)
    preprocessing = tf.keras.Model(sentences, model_inputs)

    # Individual steps of preprocessing are made available as named subobjects
    # to enable more general preprocessing. For saving, they need to be Models
    # in their own right.
    preprocessing.tokenize = tf.keras.Model(sentences, tokens)
    # Provide an equivalent to tokenize.get_special_tokens_dict().
    preprocessing.tokenize.get_special_tokens_dict = tf.train.Checkpoint()
    preprocessing.tokenize.get_special_tokens_dict.__call__ = tf.function(
        lambda: tokenize.get_special_tokens_dict(),  # pylint: disable=[unnecessary-lambda]
        input_signature=[],
    )
    if tokenize_with_offsets:
        preprocessing.tokenize_with_offsets = tf.keras.Model(sentences, [tokens, start_offsets, limit_offsets])
        preprocessing.tokenize_with_offsets.get_special_tokens_dict = preprocessing.tokenize.get_special_tokens_dict
    # Conceptually, this should be
    # preprocessing.bert_pack_inputs = tf.keras.Model(tokens, model_inputs)
    # but technicalities require us to use a wrapper (see comments there).
    # In particular, seq_length can be overridden when calling this.
    preprocessing.bert_pack_inputs = export_tfhub_lib.BertPackInputsSavedModelWrapper(pack)

    return preprocessing


if __name__ == "__main__":
    flags.DEFINE_string("model_name", "klue/bert-base", help="model name to export")
    flags.DEFINE_string("output_dir", "./models", help="output dir")
    flags.DEFINE_string("output_model_name", "klue_bert_cased", help="output model name")
    flags.DEFINE_string("tokenizer_type", "bert", help="tokenizer type")

    app.run(main)
