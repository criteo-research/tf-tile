from typing import Dict,List
from tensorflow.python.ops import math_ops
import winequality
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import tf_logging as logging
import bucketize_strategy
import tilings
import input_func
import tiled_feature_columns
import model_func
tf.logging.set_verbosity(tf.logging.INFO)

MODEL_DIR="model_dir"
numTilings=10
num_buckets = 10
batch_size = 32
example_model_fn = model_func.model_fn
tiled_feature_column_list = tiled_feature_columns.get_tiled_feature_columns(numTilings,num_buckets,winequality.FEATURES)
params={
        'feature_columns': tiled_feature_column_list,
        'hidden_units': None,
        'n_classes': winequality.get_n_classes()
        }

estimator = tf.estimator.Estimator(model_fn=example_model_fn, params=params, model_dir=MODEL_DIR)
train_spec= tf.estimator.TrainSpec(input_fn=lambda: input_func.train_input_fn(batch_size,num_buckets,numTilings),max_steps=40000)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_func.eval_input_fn(batch_size,num_buckets,numTilings),steps=100,start_delay_secs=0,throttle_secs=30)
tf.estimator.train_and_evaluate(estimator,train_spec, eval_spec)
