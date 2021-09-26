from absl import app, flags

from convert_bert import TokenizerType, convert_distilbert

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


if __name__ == "__main__":
    app.run(main)
