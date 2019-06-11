"""
Example of using LinearClassifier
"""
import logging
import os
import pwd
import getpass
import sys
import warnings
import typing
import tensorflow as tf

from functools import partial
from subprocess import check_output
from datetime import datetime

import winequality

logging.basicConfig(level="INFO")

#USER = getpass.getuser()

"""
1. Download winequality-*.csv from the Wine Quality dataset at UCI
   ML repository
   (https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
2. Upload it to HDFS
3. Pass a full URI to either of the CSV files to the example
"""
#WINE_EQUALITY_FILE = f"{packaging.get_default_fs()}/user/{USER}/tf_yarn_test/winequality-red.csv"

WINE_EQUALITY_FILE ="winequality-red.csv"
"""
Output path of the learned model on hdfs
"""
# HDFS_DIR = (f"{packaging.get_default_fs()}/user/{USER}"
#             f"/tf_yarn_test/tf_yarn_{int(datetime.now().timestamp())}")
HDFS_DIR="tmp"
train_data, test_data = winequality.get_train_eval_datasets(WINE_EQUALITY_FILE)
def train_input_fn():

    return (train_data.shuffle(1000)
            .batch(128)
            .repeat()
            .make_one_shot_iterator()
            .get_next())

def eval_input_fn():
        return (test_data.shuffle(1000)
                .batch(128)
                .make_one_shot_iterator()
                .get_next())



def experiment_fn():
    train_data, test_data = winequality.get_train_eval_datasets(WINE_EQUALITY_FILE)

    estimator = tf.estimator.LinearClassifier(
                                feature_columns=winequality.get_feature_columns(),
                                model_dir=f"{HDFS_DIR}",
                                n_classes=winequality.get_n_classes())
    train_spec= tf.estimator.TrainSpec(train_input_fn, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn,steps=100,start_delay_secs=0,throttle_secs=30)

    return estimator,train_spec, eval_spec
        


if __name__ == "__main__":
    # iterator = train_input_fn()
    # sess = tf.Session()
    # for i in range(10):
    #     print(sess.run(iterator))
    estimator,train_spec, eval_spec = experiment_fn()
    tf.estimator.train_and_evaluate(estimator,train_spec, eval_spec)
    
    

   
