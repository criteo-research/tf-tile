import logging
import argparse
import os
import sys
import time
from datetime import datetime
import tensorflow as tf
#python_root = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "../main/python")
#sys.path.append(python_root)
#from atf import experiment_fn  # noqa
#from atf import Config  # noqa
#from atf import experiment_fn


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    #args = get_cli_args()

    model_dir="tmp"
    model_dir = f"model_dir"

    config = Config.Config(
        io=Config.IOConfig(
            training_dir=args.training_dir,
            validation_dir=args.validation_dir,
            model_path=model_dir
        ),
        training=Config.TrainingConfig(
            model=args.model,
            max_steps=args.training_steps,
            batch_size=args.batch_size,
            dropout=args.dropout,
            use_bag_cross=args.use_bag_cross,
            use_batch_norm=(not args.no_batch_norm)
        ),
        evaluation=Config.EvaluationConfig(
            steps=args.evaluation_steps,
            batch_size=args.test_batch_size,
            save_checkpoints_steps=args.save_checkpoints_steps,
            keep_checkpoint_max=args.keep_checkpoint_max
        )
    )

    logging.info(f"Configuration: {config}")
    def get_filenames(root_dir):
        logging.info("Using rootdir='{}'".format(root_dir))
        return [os.path.join(root_dir, filepath)
                for filepath in tf.gfile.ListDirectory(root_dir)]

    print("training directory:",  config.io.training_dir)

    estimator, train_spec, val_spec = experiment_fn.get(config, args.model)
    tf.estimator.train_and_evaluate(estimator, train_spec, val_spec)
