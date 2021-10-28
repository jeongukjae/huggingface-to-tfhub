import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # noqa
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string("encoder", "models/distilbert_en_cased_L-6_H-768_A-12", help="encoder to test")
flags.DEFINE_string("preprocess", "models/distilbert_en_cased_preprocess", help="preprocess to test")


def main(argv):
    logging.info(f"Loading encoder & decoder. ({FLAGS.encoder}, {FLAGS.preprocess})")
    encoder = hub.load(FLAGS.encoder)
    preprocess = hub.load(FLAGS.preprocess)

    def _forward_pass_and_print(is_single_segment, seq_length, trainable):
        logging.info("Testing forward pass")
        logging.info(f"is_single_segment?: {is_single_segment}, seq_length: {seq_length}, trainable: {trainable}")
        encoder_model = hub.KerasLayer(encoder)
        preprocess_args = dict()
        if seq_length:
            preprocess_args["seq_length"] = seq_length

        if is_single_segment:
            inputs = tf.keras.Input([], dtype=tf.string)
            preprocess_model = hub.KerasLayer(preprocess)
            encoder_inputs = preprocess_model(inputs)
        else:
            inputs = [tf.keras.Input([], dtype=tf.string), tf.keras.Input([], dtype=tf.string)]
            tokenize = hub.KerasLayer(preprocess.tokenize)
            bert_pack_inputs = hub.KerasLayer(preprocess.bert_pack_inputs, arguments=preprocess_args)

            tokenized_inputs = [tokenize(segment) for segment in inputs]
            encoder_inputs = bert_pack_inputs(tokenized_inputs)

        logging.info(f"Text inputs: {inputs}, Encoder inputs: {encoder_inputs}")
        encoder_outputs = encoder_model(encoder_inputs)
        logging.info(f"Encoder outputs: {encoder_outputs}")
        model = tf.keras.Model(inputs, encoder_outputs)
        model.summary()

    for is_single_segment in (True, False):
        for seq_length in (30, None):
            for trainable in (True, None):
                if is_single_segment and seq_length is not None:
                    continue

                _forward_pass_and_print(is_single_segment, seq_length, trainable)


if __name__ == "__main__":
    app.run(main)
