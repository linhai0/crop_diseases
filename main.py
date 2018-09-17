import os, json
import tensorflow as tf
from config import FLAGS
# See the __init__ script in the models folder
# `make_models` is a helper function to load any models you have
from models import make_models
# from hpsearch import hyperband, randomsearch
from models.model_train import train
from models.simple_net import sim_net


def main(_):
    # config = FLAGS.FLAGS.__flags.copy()
    # fixed_params must be a string to be passed in the shell, let's use JSON
    # config["fixed_params"] = json.loads(config["fixed_params"])

    if FLAGS.FLAGS.fullsearch:
        # Some code for HP search ...
        pass
    else:
        model = make_models(FLAGS.FLAGS)

        if FLAGS.FLAGS.infer:
            # Some code for inference ...
            pass
        else:
            train(model, FLAGS.FLAGS)


if __name__ == '__main__':
    tf.app.run()
