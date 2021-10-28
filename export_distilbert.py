import collections
import enum
import os
from typing import Any, Dict, List, Optional, Union
from urllib import request as urllib_request

import numpy as np
import tensorflow as tf
import tensorflow_text as text
import torch
from absl import app, flags, logging
from official.nlp.keras_nlp import layers as keras_layers
from official.nlp.modeling import layers
from official.nlp.modeling.layers import text_layers
from transformers import DistilBertModel

from utils import BertPackInputsSavedModelWrapper, get_activation, get_config, get_tokenizer_config

FLAGS = flags.FLAGS
flags.DEFINE_string("model_name", "distilbert-base-uncased", help="model name to export")
flags.DEFINE_string("output_dir", "./models", help="output dir")
flags.DEFINE_string("output_model_name", "distilbert_en_uncased", help="output model name")
flags.DEFINE_string("tokenizer_type", "bert", help="tokenizer type")


def main(argv):
    convert_distilbert(
        model_name=FLAGS.model_name,
        output_dir=FLAGS.output_dir,
        output_model_name=FLAGS.output_model_name,
        tokenizer_type=TokenizerType(FLAGS.tokenizer_type),
    )


class TokenizerType(enum.Enum):
    BERT = "bert"
    SENTENCEPIECE = "spm"


def convert_distilbert(
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
                tf.reshape(
                    state_dict[f"transformer.layer.{layer_index}.attention.q_lin.weight"].T,
                    [config["dim"], config["n_heads"], -1],
                ),
                tf.reshape(
                    state_dict[f"transformer.layer.{layer_index}.attention.q_lin.bias"],
                    [config["n_heads"], -1],
                ),
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer._key_dense.set_weights(
            [
                tf.reshape(
                    state_dict[f"transformer.layer.{layer_index}.attention.k_lin.weight"].T,
                    [config["dim"], config["n_heads"], -1],
                ),
                tf.reshape(
                    state_dict[f"transformer.layer.{layer_index}.attention.k_lin.bias"],
                    [config["n_heads"], -1],
                ),
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer._value_dense.set_weights(
            [
                tf.reshape(
                    state_dict[f"transformer.layer.{layer_index}.attention.v_lin.weight"].T,
                    [config["dim"], config["n_heads"], -1],
                ),
                tf.reshape(
                    state_dict[f"transformer.layer.{layer_index}.attention.v_lin.bias"],
                    [config["n_heads"], -1],
                ),
            ]
        )
        encoder._transformer_layers[layer_index]._attention_layer._output_dense.set_weights(
            [
                tf.reshape(
                    state_dict[f"transformer.layer.{layer_index}.attention.out_lin.weight"].T,
                    [config["n_heads"], -1, config["dim"]],
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

    with torch.no_grad():
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
    model.save(
        os.path.join(
            output_dir,
            output_model_name + f"_L-{config['n_layers']}_H-{config['dim']}_A-{config['n_heads']}",
        )
    )

    if tokenizer_type == TokenizerType.BERT:
        vocab_file = os.path.join(temp_dir, "vocab.txt")
        urllib_request.urlretrieve(f"https://huggingface.co/{model_name}/resolve/main/vocab.txt", vocab_file)
        tokenizer_config = get_tokenizer_config(model_name)
        do_lower_case = tokenizer_config["do_lower_case"] if "do_lower_case" in tokenizer_config else False

        logging.info(f"do_lower_case: {do_lower_case}")
        preprocessor = create_distilbert_preprocessing(vocab_file=vocab_file, do_lower_case=do_lower_case)
        preprocessor.save(os.path.join(output_dir, output_model_name + "_preprocess"))


@tf.keras.utils.register_keras_serializable(package="covnerting_tf")
class DistilBertEncoder(tf.keras.Model):
    """DistilBertEncoder.

    `type_embedding_layer` and `pooler_layer` is deleted.

    This class is modified version of BertEncoder in tensorflow/models to support
    DistilBertEncoder.

    *Note* that the network is constructed by
    [Keras Functional API](https://keras.io/guides/functional_api/).

    Args:
      vocab_size: The size of the token vocabulary.
      hidden_size: The size of the transformer hidden layers.
      num_layers: The number of transformer layers.
      num_attention_heads: The number of attention heads for each transformer. The
        hidden size must be divisible by the number of attention heads.
      max_sequence_length: The maximum sequence length that this encoder can
        consume. If None, max_sequence_length uses the value from sequence length.
        This determines the variable shape for positional embeddings.
      inner_dim: The output dimension of the first Dense layer in a two-layer
          feedforward network for each transformer.
      inner_activation: The activation for the first Dense layer in a two-layer
          feedforward network for each transformer.
      output_dropout: Dropout probability for the post-attention and output
          dropout.
      attention_dropout: The dropout rate to use for the attention layers
        within the transformer layers.
      initializer: The initialzer to use for all weights in this encoder.
      output_range: The sequence output range, [0, output_range), by slicing the
        target sequence of the last transformer layer. `None` means the entire
        target sequence will attend to the source sequence, which yields the full
        output.
      embedding_width: The width of the word embeddings. If the embedding width is
        not equal to hidden size, embedding parameters will be factorized into two
        matrices in the shape of ['vocab_size', 'embedding_width'] and
        ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
        smaller than 'hidden_size').
      embedding_layer: An optional Layer instance which will be called to
       generate embeddings for the input word IDs.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        max_sequence_length=512,
        inner_dim=3072,
        inner_activation=lambda x: tf.keras.activations.gelu(x, approximate=True),
        output_dropout=0.1,
        attention_dropout=0.1,
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        output_range=None,
        embedding_width=None,
        embedding_layer=None,
        **kwargs,
    ):
        activation = tf.keras.activations.get(inner_activation)
        initializer = tf.keras.initializers.get(initializer)

        word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_word_ids")
        mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_mask")

        if embedding_width is None:
            embedding_width = hidden_size

        if embedding_layer is None:
            embedding_layer_inst = keras_layers.OnDeviceEmbedding(
                vocab_size=vocab_size,
                embedding_width=embedding_width,
                initializer=initializer,
                name="word_embeddings",
            )
        else:
            embedding_layer_inst = embedding_layer
        word_embeddings = embedding_layer_inst(word_ids)

        # Always uses dynamic slicing for simplicity.
        position_embedding_layer = keras_layers.PositionEmbedding(
            initializer=initializer,
            max_length=max_sequence_length,
            name="position_embedding",
        )
        position_embeddings = position_embedding_layer(word_embeddings)

        embeddings = tf.keras.layers.Add()([word_embeddings, position_embeddings])

        embedding_norm_layer = tf.keras.layers.LayerNormalization(name="embeddings/layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)

        embeddings = embedding_norm_layer(embeddings)
        embeddings = tf.keras.layers.Dropout(rate=output_dropout)(embeddings)

        # We project the 'embedding' output to 'hidden_size' if it is not already
        # 'hidden_size'.
        if embedding_width != hidden_size:
            embedding_projection = tf.keras.layers.experimental.EinsumDense(
                "...x,xy->...y",
                output_shape=hidden_size,
                bias_axes="y",
                kernel_initializer=initializer,
                name="embedding_projection",
            )
            embeddings = embedding_projection(embeddings)
        else:
            embedding_projection = None

        transformer_layers = []
        data = embeddings
        attention_mask = keras_layers.SelfAttentionMask()(data, mask)
        encoder_outputs = []
        for i in range(num_layers):
            if i == num_layers - 1 and output_range is not None:
                transformer_output_range = output_range
            else:
                transformer_output_range = None
            layer = keras_layers.TransformerEncoderBlock(
                num_attention_heads=num_attention_heads,
                inner_dim=inner_dim,
                inner_activation=inner_activation,
                output_dropout=output_dropout,
                attention_dropout=attention_dropout,
                output_range=transformer_output_range,
                kernel_initializer=initializer,
                name="transformer/layer_%d" % i,
            )
            transformer_layers.append(layer)
            data = layer([data, attention_mask])
            encoder_outputs.append(data)

        last_encoder_output = encoder_outputs[-1]
        # Applying a tf.slice op (through subscript notation) to a Keras tensor
        # like this will create a SliceOpLambda layer. This is better than a Lambda
        # layer with Python code, because that is fundamentally less portable.
        first_token_tensor = last_encoder_output[:, 0, :]
        cls_output = first_token_tensor

        outputs = dict(
            sequence_output=encoder_outputs[-1],
            pooled_output=cls_output,
            encoder_outputs=encoder_outputs,
        )

        # Once we've created the network using the Functional API, we call
        # super().__init__ as though we were invoking the Functional API Model
        # constructor, resulting in this object having all the properties of a model
        # created using the Functional API. Once super().__init__ is called, we
        # can assign attributes to `self` - note that all `self` assignments are
        # below this line.
        super(DistilBertEncoder, self).__init__(inputs=[word_ids, mask], outputs=outputs, **kwargs)

        config_dict = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "max_sequence_length": max_sequence_length,
            "inner_dim": inner_dim,
            "inner_activation": tf.keras.activations.serialize(activation),
            "output_dropout": output_dropout,
            "attention_dropout": attention_dropout,
            "initializer": tf.keras.initializers.serialize(initializer),
            "output_range": output_range,
            "embedding_width": embedding_width,
            "embedding_layer": embedding_layer,
        }

        # We are storing the config dict as a namedtuple here to ensure checkpoint
        # compatibility with an earlier version of this model which did not track
        # the config dict attribute. TF does not track immutable attrs which
        # do not contain Trackables, so by creating a config namedtuple instead of
        # a dict we avoid tracking it.
        config_cls = collections.namedtuple("Config", config_dict.keys())
        self._config = config_cls(**config_dict)
        self._transformer_layers = transformer_layers
        self._embedding_norm_layer = embedding_norm_layer
        self._embedding_layer = embedding_layer_inst
        self._position_embedding_layer = position_embedding_layer
        if embedding_projection is not None:
            self._embedding_projection = embedding_projection

    def get_embedding_table(self):
        return self._embedding_layer.embeddings

    def get_embedding_layer(self):
        return self._embedding_layer

    def get_config(self):
        return dict(self._config._asdict())

    @property
    def transformer_layers(self):
        """List of Transformer layers in the encoder."""
        return self._transformer_layers

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if "embedding_layer" in config and config["embedding_layer"] is not None:
            warn_string = (
                "You are reloading a model that was saved with a "
                "potentially-shared embedding layer object. If you contine to "
                "train this model, the embedding layer will no longer be shared. "
                "To work around this, load the model outside of the Keras API."
            )
            print("WARNING: " + warn_string)
            logging.warn(warn_string)

        return cls(**config)


def create_distilbert_preprocessing(
    *,
    do_lower_case: bool,
    vocab_file: Optional[str] = None,
    sp_model_file: Optional[str] = None,
    tokenize_with_offsets: bool = True,
    default_seq_length: int = 128,
) -> tf.keras.Model:
    """Returns a preprocessing Model for given tokenization parameters.

    This function builds a Keras Model with attached subobjects suitable for
    saving to a SavedModel. The resulting SavedModel implements the Preprocessor
    API for Text embeddings with Transformer Encoders described at
    https://www.tensorflow.org/hub/common_saved_model_apis/text.

    Args:
      vocab_file: The path to the wordpiece vocab file, or None.
      sp_model_file: The path to the sentencepiece model file, or None.
        Exactly one of vocab_file and sp_model_file must be set.
        This determines the type of tokenzer that is used.
      do_lower_case: Whether to do lower case.
      tokenize_with_offsets: Whether to include the .tokenize_with_offsets
        subobject.
      default_seq_length: The sequence length of preprocessing results from
        root callable. This is also the default sequence length for the
        bert_pack_inputs subobject.

    Returns:
      A tf.keras.Model object with several attached subobjects, suitable for
      saving as a preprocessing SavedModel.
    """
    # Select tokenizer.
    if bool(vocab_file) == bool(sp_model_file):
        raise ValueError("Must set exactly one of vocab_file, sp_model_file")
    if vocab_file:
        tokenize = layers.BertTokenizer(
            vocab_file=vocab_file,
            lower_case=do_lower_case,
            tokenize_with_offsets=tokenize_with_offsets,
        )
    else:
        tokenize = layers.SentencepieceTokenizer(
            model_file_path=sp_model_file,
            lower_case=do_lower_case,
            strip_diacritics=True,  # Strip diacritics to follow ALBERT model.
            tokenize_with_offsets=tokenize_with_offsets,
        )

    # The root object of the preprocessing model can be called to do
    # one-shot preprocessing for users with single-sentence inputs.
    sentences = tf.keras.layers.Input(shape=(), dtype=tf.string, name="sentences")
    if tokenize_with_offsets:
        tokens, start_offsets, limit_offsets = tokenize(sentences)
    else:
        tokens = tokenize(sentences)
    pack = DistilBertPackInputs(
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
    preprocessing.bert_pack_inputs = BertPackInputsSavedModelWrapper(pack, pack.bert_pack_inputs)

    return preprocessing


class DistilBertPackInputs(tf.keras.layers.Layer):
    """Packs tokens into model inputs for BERT."""

    def __init__(
        self,
        seq_length,
        *,
        start_of_sequence_id=None,
        end_of_segment_id=None,
        padding_id=None,
        special_tokens_dict=None,
        truncator="round_robin",
        **kwargs,
    ):
        """Initializes with a target `seq_length`, relevant token ids and truncator.

        Args:
          seq_length: The desired output length. Must not exceed the max_seq_length
            that was fixed at training time for the BERT model receiving the inputs.
          start_of_sequence_id: The numeric id of the token that is to be placed
            at the start of each sequence (called "[CLS]" for BERT).
          end_of_segment_id: The numeric id of the token that is to be placed
            at the end of each input segment (called "[SEP]" for BERT).
          padding_id: The numeric id of the token that is to be placed into the
            unused positions after the last segment in the sequence
            (called "[PAD]" for BERT).
          special_tokens_dict: Optionally, a dict from Python strings to Python
            integers that contains values for `start_of_sequence_id`,
            `end_of_segment_id` and `padding_id`. (Further values in the dict are
            silenty ignored.) If this is passed, separate *_id arguments must be
            omitted.
          truncator: The algorithm to truncate a list of batched segments to fit a
            per-example length limit. The value can be either `round_robin` or
            `waterfall`:
              (1) For "round_robin" algorithm, available space is assigned
              one token at a time in a round-robin fashion to the inputs that still
              need some, until the limit is reached. It currently only supports
              one or two segments.
              (2) For "waterfall" algorithm, the allocation of the budget is done
                using a "waterfall" algorithm that allocates quota in a
                left-to-right manner and fills up the buckets until we run out of
                budget. It support arbitrary number of segments.

          **kwargs: standard arguments to `Layer()`.

        Raises:
          ImportError: if importing `tensorflow_text` failed.
        """
        super().__init__(**kwargs)
        self.seq_length = seq_length
        if truncator not in ("round_robin", "waterfall"):
            raise ValueError("Only 'round_robin' and 'waterfall' algorithms are " "supported, but got %s" % truncator)
        self.truncator = truncator
        self._init_token_ids(
            start_of_sequence_id=start_of_sequence_id,
            end_of_segment_id=end_of_segment_id,
            padding_id=padding_id,
            special_tokens_dict=special_tokens_dict,
        )

    def _init_token_ids(
        self,
        *,
        start_of_sequence_id,
        end_of_segment_id,
        padding_id,
        special_tokens_dict,
    ):
        usage = "Must pass either all of start_of_sequence_id, end_of_segment_id, " "padding_id as arguments, or else a special_tokens_dict " "with those keys."
        special_tokens_args = [start_of_sequence_id, end_of_segment_id, padding_id]
        if special_tokens_dict is None:
            if any(x is None for x in special_tokens_args):
                return ValueError(usage)
            self.start_of_sequence_id = int(start_of_sequence_id)
            self.end_of_segment_id = int(end_of_segment_id)
            self.padding_id = int(padding_id)
        else:
            if any(x is not None for x in special_tokens_args):
                return ValueError(usage)
            self.start_of_sequence_id = int(special_tokens_dict["start_of_sequence_id"])
            self.end_of_segment_id = int(special_tokens_dict["end_of_segment_id"])
            self.padding_id = int(special_tokens_dict["padding_id"])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["seq_length"] = self.seq_length
        config["start_of_sequence_id"] = self.start_of_sequence_id
        config["end_of_segment_id"] = self.end_of_segment_id
        config["padding_id"] = self.padding_id
        config["truncator"] = self.truncator
        return config

    def call(self, inputs: Union[tf.RaggedTensor, List[tf.RaggedTensor]]):
        """Adds special tokens to pack a list of segments into BERT input Tensors.

        Args:
          inputs: A Python list of one or two RaggedTensors, each with the batched
            values one input segment. The j-th segment of the i-th input example
            consists of slice `inputs[j][i, ...]`.

        Returns:
          A nest of Tensors for use as input to the BERT TransformerEncoder.
        """
        # DistilBertPackInputsSavedModelWrapper relies on only calling bert_pack_inputs()
        return DistilBertPackInputs.bert_pack_inputs(
            inputs,
            self.seq_length,
            start_of_sequence_id=self.start_of_sequence_id,
            end_of_segment_id=self.end_of_segment_id,
            padding_id=self.padding_id,
            truncator=self.truncator,
        )

    @staticmethod
    def bert_pack_inputs(
        inputs: Union[tf.RaggedTensor, List[tf.RaggedTensor]],
        seq_length: Union[int, tf.Tensor],
        start_of_sequence_id: Union[int, tf.Tensor],
        end_of_segment_id: Union[int, tf.Tensor],
        padding_id: Union[int, tf.Tensor],
        truncator="round_robin",
    ):
        """Freestanding equivalent of the DistilBertPackInputs layer."""
        # Sanitize inputs.
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not inputs:
            raise ValueError("At least one input is required for packing")
        input_ranks = [rt.shape.rank for rt in inputs]
        if None in input_ranks or len(set(input_ranks)) > 1:
            raise ValueError("All inputs for packing must have the same known rank, " "found ranks " + ",".join(input_ranks))
        # Flatten inputs to [batch_size, (tokens)].
        if input_ranks[0] > 2:
            inputs = [rt.merge_dims(1, -1) for rt in inputs]
        # In case inputs weren't truncated (as they should have been),
        # fall back to some ad-hoc truncation.
        num_special_tokens = len(inputs) + 1
        if truncator == "round_robin":
            trimmed_segments = text_layers.round_robin_truncate_inputs(inputs, seq_length - num_special_tokens)
        elif truncator == "waterfall":
            trimmed_segments = text.WaterfallTrimmer(seq_length - num_special_tokens).trim(inputs)
        else:
            raise ValueError("Unsupported truncator: %s" % truncator)
        # Combine segments.
        segments_combined, segment_ids = text.combine_segments(
            trimmed_segments,
            start_of_sequence_id=start_of_sequence_id,
            end_of_segment_id=end_of_segment_id,
        )
        # Pad to dense Tensors.
        input_word_ids, _ = text.pad_model_inputs(segments_combined, seq_length, pad_value=padding_id)
        _, input_mask = text.pad_model_inputs(segment_ids, seq_length, pad_value=0)
        # Work around broken shape inference.
        output_shape = tf.stack([inputs[0].nrows(out_type=tf.int32), tf.cast(seq_length, dtype=tf.int32)])  # batch_size

        def _reshape(t):
            return tf.reshape(t, output_shape)

        # Assemble nest of input tensors as expected by BERT TransformerEncoder.
        return dict(input_word_ids=_reshape(input_word_ids), input_mask=_reshape(input_mask))


if __name__ == "__main__":
    app.run(main)
