import collections
import enum
import os
from typing import Any, Dict, List, Optional, Union
from urllib import request as urllib_request

import numpy as np
import sentencepiece.sentencepiece_model_pb2 as sp_model_pb2
import tensorflow as tf
import tensorflow_text as text
import torch
from absl import app, flags, logging
from official.nlp.keras_nlp import layers as keras_layers
from official.nlp.modeling import layers
from transformers import RobertaModel

from utils import BertPackInputsSavedModelWrapper, get_activation, get_config, get_tokenizer_config

FLAGS = flags.FLAGS


def main(argv):
    convert_roberta(
        model_name=FLAGS.model_name,
        output_dir=FLAGS.output_dir,
        output_model_name=FLAGS.output_model_name,
        tokenizer_type=TokenizerType(FLAGS.tokenizer_type),
    )


class TokenizerType(enum.Enum):
    BPE = "bpe"
    XLM_SPM = "xlm-spm"
    KLUE = "klue"


def convert_roberta(
    model_name: str,
    output_dir: str,
    output_model_name: str,
    tokenizer_type: TokenizerType = TokenizerType.BPE,
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
    position_id_offset = config["pad_token_id"] + 1
    encoder = RobertaEncoder(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_hidden_layers"],
        type_vocab_size=config["type_vocab_size"],
        num_attention_heads=config["num_attention_heads"],
        layer_norm_eps=config["layer_norm_eps"],
        max_sequence_length=config["max_position_embeddings"] - position_id_offset,
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
    torch_model = RobertaModel.from_pretrained(model_name)
    state_dict = torch_model.state_dict()

    logging.info("Convert weights")
    encoder._embedding_layer.set_weights([state_dict["embeddings.word_embeddings.weight"]])
    encoder._position_embedding_layer.set_weights([state_dict["embeddings.position_embeddings.weight"][position_id_offset:]])
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

    if tokenizer_type == TokenizerType.BPE:
        tokenizer_config = get_tokenizer_config(model_name, filename="tokenizer.json")
        tokenizer_config["model"]["vocab"] = {k.replace("Ġ", "▁"): v for k, v in tokenizer_config["model"]["vocab"].items()}
        tokenizer_config["model"]["merges"] = [merge.replace("Ġ", "▁") for merge in tokenizer_config["model"]["merges"]]
        preprocessor = create_roberta_preprocessing(tokenizer_type, bpe_tokenizer_config=tokenizer_config)
        preprocessor.save(os.path.join(output_dir, output_model_name + "_preprocess"))
    elif tokenizer_type == TokenizerType.XLM_SPM:
        tokenizer_config = get_tokenizer_config(model_name, filename="tokenizer.json")
        spm_file = os.path.join(temp_dir, "sentencepiece.bpe.model")
        urllib_request.urlretrieve(f"https://huggingface.co/{model_name}/resolve/main/sentencepiece.bpe.model", spm_file)
        preprocessor = create_roberta_preprocessing(tokenizer_type, sp_model_file=spm_file, vocabs=[tokenizer_config["model"]["vocab"][index][0] for index in range(config["vocab_size"])])
        preprocessor.save(os.path.join(output_dir, output_model_name + "_preprocess"))
    elif tokenizer_type == TokenizerType.KLUE:
        vocab_file = os.path.join(temp_dir, "vocab.txt")
        urllib_request.urlretrieve(f"https://huggingface.co/{model_name}/resolve/main/vocab.txt", vocab_file)
        tokenizer_config = get_tokenizer_config(model_name)
        do_lower_case = tokenizer_config["do_lower_case"] if "do_lower_case" in tokenizer_config else False

        logging.info(f"do_lower_case: {do_lower_case}")
        preprocessor = create_roberta_preprocessing(tokenizer_type, vocab_file=vocab_file, do_lower_case=do_lower_case)
        preprocessor.save(os.path.join(output_dir, output_model_name + "_preprocess"))
    else:
        raise ValueError


@tf.keras.utils.register_keras_serializable(package="covnerting_tf")
class RobertaEncoder(tf.keras.Model):
    """RoBERTa encoder

    I added layer_norm_eps

    Bi-directional Transformer-based encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    The default values for this object are taken from the BERT-Base implementation
    in "BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding".

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
      type_vocab_size: The number of types that the 'type_ids' input can take.
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
        type_vocab_size=16,
        inner_dim=3072,
        inner_activation=lambda x: tf.keras.activations.gelu(x, approximate=True),
        output_dropout=0.1,
        attention_dropout=0.1,
        layer_norm_eps=1e-12,
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
        type_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_type_ids")

        if embedding_width is None:
            embedding_width = hidden_size

        if embedding_layer is None:
            embedding_layer_inst = keras_layers.OnDeviceEmbedding(vocab_size=vocab_size, embedding_width=embedding_width, initializer=initializer, name="word_embeddings")
        else:
            embedding_layer_inst = embedding_layer
        word_embeddings = embedding_layer_inst(word_ids)

        # Always uses dynamic slicing for simplicity.
        position_embedding_layer = keras_layers.PositionEmbedding(initializer=initializer, max_length=max_sequence_length, name="position_embedding")
        position_embeddings = position_embedding_layer(word_embeddings)
        type_embedding_layer = keras_layers.OnDeviceEmbedding(vocab_size=type_vocab_size, embedding_width=embedding_width, initializer=initializer, use_one_hot=True, name="type_embeddings")
        type_embeddings = type_embedding_layer(type_ids)

        embeddings = tf.keras.layers.Add()([word_embeddings, position_embeddings, type_embeddings])

        embedding_norm_layer = tf.keras.layers.LayerNormalization(name="embeddings/layer_norm", axis=-1, epsilon=layer_norm_eps, dtype=tf.float32)

        embeddings = embedding_norm_layer(embeddings)
        embeddings = tf.keras.layers.Dropout(rate=output_dropout)(embeddings)

        # We project the 'embedding' output to 'hidden_size' if it is not already
        # 'hidden_size'.
        if embedding_width != hidden_size:
            embedding_projection = tf.keras.layers.experimental.EinsumDense("...x,xy->...y", output_shape=hidden_size, bias_axes="y", kernel_initializer=initializer, name="embedding_projection")
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
                norm_epsilon=layer_norm_eps,
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
        pooler_layer = tf.keras.layers.Dense(units=hidden_size, activation="tanh", kernel_initializer=initializer, name="pooler_transform")
        cls_output = pooler_layer(first_token_tensor)

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
        super(RobertaEncoder, self).__init__(inputs=[word_ids, mask, type_ids], outputs=outputs, **kwargs)

        config_dict = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "max_sequence_length": max_sequence_length,
            "type_vocab_size": type_vocab_size,
            "inner_dim": inner_dim,
            "inner_activation": tf.keras.activations.serialize(activation),
            "output_dropout": output_dropout,
            "layer_norm_eps": layer_norm_eps,
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
        self._pooler_layer = pooler_layer
        self._transformer_layers = transformer_layers
        self._embedding_norm_layer = embedding_norm_layer
        self._embedding_layer = embedding_layer_inst
        self._position_embedding_layer = position_embedding_layer
        self._type_embedding_layer = type_embedding_layer
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

    @property
    def pooler_layer(self):
        """The pooler dense layer after the transformer layers."""
        return self._pooler_layer

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


def create_roberta_preprocessing(
    tokenizer_type: TokenizerType,
    *,
    vocab_file: str = None,
    do_lower_case: bool = False,
    bpe_tokenizer_config: Optional[dict] = None,
    vocabs: Optional[List] = None,
    sp_model_file: Optional[str] = None,
    tokenize_with_offsets: bool = True,
    default_seq_length: int = 128,
) -> tf.keras.Model:
    # Select tokenizer.
    if tokenizer_type == TokenizerType.BPE:
        assert bpe_tokenizer_config
        tokenize = BPESentencepieceTokenizer(
            model_serialized_proto=_bpe_to_sentencepiece_proto(tokenizer_config=bpe_tokenizer_config),
            lower_case=False,
            vocabs=bpe_tokenizer_config["model"]["vocab"],
            tokenize_with_offsets=True,
            strip_diacritics=False,
        )
    elif tokenizer_type == TokenizerType.XLM_SPM:
        assert vocabs
        tokenize = XLMSentencepieceTokenizer(
            model_file_path=sp_model_file,
            vocabs=vocabs,
            lower_case=do_lower_case,
            tokenize_with_offsets=tokenize_with_offsets,
        )
    elif tokenizer_type == TokenizerType.KLUE:
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
    pack = RobertaPackInputs(
        seq_length=default_seq_length,
        special_tokens_dict=tokenize.get_special_tokens_dict(),
        double_sep_token_between_segments=tokenizer_type != TokenizerType.KLUE,
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
    preprocessing.bert_pack_inputs = BertPackInputsSavedModelWrapper(pack, pack.roberta_pack_inputs)

    return preprocessing


class XLMSentencepieceTokenizer(layers.SentencepieceTokenizer):
    def __init__(self, *args, vocabs=[], **kwargs):
        self._vocab_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(vocabs),
                values=tf.range(len(vocabs)),
            ),
            default_value=vocabs.index("<unk>"),
        )
        self._inverse_vocab_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.range(len(vocabs)),
                values=tf.constant(vocabs),
            ),
            default_value="<unk>",
        )

        super().__init__(*args, **kwargs)

    def _create_tokenizer(self):
        model = sp_model_pb2.ModelProto()
        model.ParseFromString(self._model_serialized_proto)
        is_mask_token_exists = any([p.piece == "<mask>" for p in model.pieces])
        if not is_mask_token_exists:
            logging.info("Adding mask token")
            new_piece = sp_model_pb2.ModelProto.SentencePiece()
            new_piece.piece = "<mask>"
            new_piece.score = 0.0
            new_piece.type = sp_model_pb2.ModelProto.SentencePiece.Type.USER_DEFINED
            model.pieces.insert(3, new_piece)
            self._model_serialized_proto = model.SerializeToString()

        return text.SentencepieceTokenizer(
            model=self._model_serialized_proto,
            out_type=tf.string,
            nbest_size=self._nbest_size,
            alpha=self._alpha,
        )

    def _create_special_tokens_dict(self):
        special_tokens = dict(start_of_sequence_id=b"<s>", end_of_segment_id=b"</s>", padding_id=b"<pad>", mask_id=b"<mask>")
        with tf.init_scope():
            special_token_ids = self._vocab_table.lookup(tf.constant(list(special_tokens.values()), tf.string))
            inverse_tokens = self._inverse_vocab_table.lookup(special_token_ids)
            vocab_size = self._tokenizer.vocab_size()
        result = dict(vocab_size=int(vocab_size))  # Numpy to Python.
        for name, token_id, inverse_token in zip(special_tokens, special_token_ids, inverse_tokens):
            if special_tokens[name] == inverse_token:
                result[name] = int(token_id)
            else:
                logging.warning('Could not find %s as token "%s" in sentencepiece model, ' 'got "%s"', name, special_tokens[name], inverse_token)
        return result

    def call(self, inputs):
        tokens = super().call(inputs)
        if self.tokenize_with_offsets:
            return self._vocab_table.lookup(tokens[0]), tokens[1], tokens[2]
        else:
            return self._vocab_table.lookup(tokens)


class BPESentencepieceTokenizer(layers.SentencepieceTokenizer):
    def __init__(
        self,
        *args,
        vocabs={},
        unk_token="<unk>",
        **kwargs,
    ):
        bpe_tokens = sorted(_bytes_to_unicode().items())
        self._bpe_tokens = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant([c[0] for c in bpe_tokens]),
                values=tf.constant([c[1] for c in bpe_tokens]),
            ),
            default_value=bpe_tokens[0][1],
        )
        unk_id = vocabs[unk_token]
        inverse_vocab = {v: k for k, v in vocabs.items()}
        vocabs = [inverse_vocab[i] for i in range(len(vocabs))]
        self._vocab_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(vocabs),
                values=tf.range(len(vocabs)),
            ),
            default_value=unk_id,
        )
        self._inverse_vocab_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.range(len(vocabs)),
                values=tf.constant(vocabs),
            ),
            default_value=unk_token,
        )

        super().__init__(*args, **kwargs)

    def _create_tokenizer(self):
        return text.SentencepieceTokenizer(
            model=self._model_serialized_proto,
            out_type=tf.string,
            nbest_size=self._nbest_size,
            alpha=self._alpha,
        )

    def call(self, inputs):
        regex_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"""
        inputs = text.regex_split(inputs, delim_regex_pattern=regex_pattern, keep_delim_regex_pattern=regex_pattern)

        def prepare_bpe(x):
            x = tf.map_fn(lambda y: tf.cast(tf.io.decode_raw(y, out_type=tf.uint8), tf.int32), x, fn_output_signature=tf.RaggedTensorSpec([None], dtype=tf.int32))
            x = self._bpe_tokens.lookup(x)
            x = tf.strings.reduce_join(x, axis=-1)
            return x

        inputs = tf.map_fn(
            lambda x: prepare_bpe(x),
            inputs,
            fn_output_signature=tf.RaggedTensorSpec([None], dtype=tf.string),
        )
        if self.tokenize_with_offsets:
            tokens, starts, ends = super().call(inputs.flat_values)
            tokens = inputs.with_flat_values(tokens).merge_dims(-2, -1)
            return self._vocab_table.lookup(tokens), starts, ends
        else:
            tokens = super().call(inputs.flat_values)
            tokens = inputs.with_flat_values(tokens).merge_dims(-2, -1)
            return self._vocab_table.lookup(tokens)

    def _create_special_tokens_dict(self):
        special_tokens = dict(start_of_sequence_id=b"<s>", end_of_segment_id=b"</s>", padding_id=b"<pad>", mask_id=b"<mask>")
        with tf.init_scope():
            special_token_ids = self._vocab_table.lookup(tf.constant(list(special_tokens.values()), tf.string))
            inverse_tokens = self._inverse_vocab_table.lookup(special_token_ids)
            vocab_size = self._vocab_table.size()
        result = dict(vocab_size=int(vocab_size))  # Numpy to Python.
        for name, token_id, inverse_token in zip(special_tokens, special_token_ids, inverse_tokens):
            if special_tokens[name] == inverse_token:
                result[name] = int(token_id)
            else:
                logging.warning('Could not find %s as token "%s" in sentencepiece model, ' 'got "%s"', name, special_tokens[name], inverse_token)
        return result


class RobertaPackInputs(tf.keras.layers.Layer):
    def __init__(
        self,
        seq_length,
        *,
        start_of_sequence_id=None,
        end_of_segment_id=None,
        padding_id=None,
        special_tokens_dict=None,
        truncator="round_robin",
        double_sep_token_between_segments=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        if truncator not in ("round_robin", "waterfall"):
            raise ValueError("Only 'round_robin' and 'waterfall' algorithms are " "supported, but got %s" % truncator)
        self.truncator = truncator
        self._init_token_ids(start_of_sequence_id=start_of_sequence_id, end_of_segment_id=end_of_segment_id, padding_id=padding_id, special_tokens_dict=special_tokens_dict)
        self._double_sep_token_between_segments = double_sep_token_between_segments

    def _init_token_ids(self, *, start_of_sequence_id, end_of_segment_id, padding_id, special_tokens_dict):
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
        config["double_sep_token_between_segments"] = self._double_sep_token_between_segments
        return config

    def call(self, inputs: Union[tf.RaggedTensor, List[tf.RaggedTensor]]):
        return self.roberta_pack_inputs(inputs, self.seq_length)

    def roberta_pack_inputs(self, inputs: Union[tf.RaggedTensor, List[tf.RaggedTensor]], seq_length: Union[int, tf.Tensor]):
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

        # For Roberta(except klue-roberta),
        # we will encode a single segment as "<s> sentence </s>",
        # and multi segments as "<s> sentence1 </s></s> sentence2 </s>"
        if self._double_sep_token_between_segments:
            num_special_tokens = len(inputs) * 2
        else:
            num_special_tokens = len(inputs) + 1
        if self.truncator == "round_robin":
            trimmed_segments = text.RoundRobinTrimmer(seq_length - num_special_tokens).trim(inputs)
        elif self.truncator == "waterfall":
            trimmed_segments = text.WaterfallTrimmer(seq_length - num_special_tokens).trim(inputs)
        else:
            raise ValueError("Unsupported truncator: %s" % self.truncator)

        if self._double_sep_token_between_segments:
            # We should insert two end_of_segment_id between segments
            def push_end_of_segment_id(segment):
                eos_tokens = tf.expand_dims(tf.repeat(self.end_of_segment_id, segment.nrows()), -1)
                eos_tokens = tf.cast(eos_tokens, segment.dtype)
                return tf.concat([eos_tokens, segment], axis=-1)

            trimmed_segments = [(segment if segment_index == 0 else push_end_of_segment_id(segment)) for segment_index, segment in enumerate(trimmed_segments)]

        # Combine segments.
        segments_combined, segment_ids = text.combine_segments(trimmed_segments, start_of_sequence_id=self.start_of_sequence_id, end_of_segment_id=self.end_of_segment_id)
        # Pad to dense Tensors.
        input_word_ids, _ = text.pad_model_inputs(segments_combined, seq_length, pad_value=self.padding_id)
        # TODO Because almost Roberta model has type_vocab_size 1, for now, we just make input_type_ids with tf.zeros
        input_type_ids = tf.zeros_like(input_word_ids, dtype=tf.int32)
        _, input_mask = text.pad_model_inputs(segment_ids, seq_length, pad_value=0)
        # Work around broken shape inference.
        output_shape = tf.stack([inputs[0].nrows(out_type=tf.int32), tf.cast(seq_length, dtype=tf.int32)])  # batch_size

        def _reshape(t):
            return tf.reshape(t, output_shape)

        # Assemble nest of input tensors as expected by BERT TransformerEncoder.
        return dict(input_word_ids=_reshape(input_word_ids), input_mask=_reshape(input_mask), input_type_ids=_reshape(input_type_ids))


# From https://github.com/jeongukjae/bpe-sentencepiece-conversion
def _bpe_to_sentencepiece_proto(tokenizer_config):
    def _get_piece(token, score, type_=sp_model_pb2.ModelProto.SentencePiece.Type.NORMAL):
        new_piece = sp_model_pb2.ModelProto.SentencePiece()
        new_piece.piece = token
        new_piece.score = score
        new_piece.type = type_
        return new_piece

    default_tokens = _bytes_to_unicode()

    m = sp_model_pb2.ModelProto()
    m.trainer_spec.model_type = sp_model_pb2.TrainerSpec.ModelType.BPE
    m.normalizer_spec.add_dummy_prefix = False
    m.normalizer_spec.remove_extra_whitespaces = False
    m.pieces.append(_get_piece("<unk>", 0, sp_model_pb2.ModelProto.SentencePiece.Type.UNKNOWN))
    m.pieces.append(_get_piece("<mask>", 0, sp_model_pb2.ModelProto.SentencePiece.Type.USER_DEFINED))
    for token in tokenizer_config["added_tokens"]:
        if token["content"] == "<unk>":
            continue
        m.pieces.append(_get_piece(token["content"], 0, sp_model_pb2.ModelProto.SentencePiece.Type.CONTROL))

    offset = -0.1
    for index, merge in enumerate(tokenizer_config["model"]["merges"]):
        token = merge.replace(" ", "")
        m.pieces.append(_get_piece(token, index * -0.1 + offset))

    offset += len(tokenizer_config["model"]["merges"]) * -0.1
    for index, b in enumerate(default_tokens.values()):
        m.pieces.append(_get_piece(b, index * -0.1 + offset))
    return m.SerializeToString()


# From https://huggingface.co/transformers/_modules/transformers/models/gpt2/tokenization_gpt2.html#GPT2Tokenizer
def _bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if chr(b) == " ":
            bs.append(b)
            cs.append(ord("▁"))
            n += 1
            continue

        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


if __name__ == "__main__":
    flags.DEFINE_string("model_name", "roberta-base", help="model name to export")
    flags.DEFINE_string("output_dir", "./models", help="output dir")
    flags.DEFINE_string("output_model_name", "roberta_en_uncased", help="output model name")
    flags.DEFINE_string("tokenizer_type", "bpe", help="tokenizer type")

    app.run(main)
