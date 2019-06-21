'''
Using tile-coding technique as an efficient way of sparse-coding for real-valued data
Here, winequality dataset has been used as an example 
'''
import os
import sys
import tensorflow as tf

import winequality

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from tf_tile.tile_strategy import TileStrategy
from tf_tile.tiled_feature_columns import TiledFeatureColumns
from tf_tile.tilings import Tilings

tf.logging.set_verbosity(tf.logging.INFO)

"""
Download winequality-red.csv from the Wine Quality dataset at UCI
ML repository
(https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
"""
WINE_EQUALITY_FILE = os.path.join(os.path.dirname(__file__), "winequality-red.csv")

def get_input_fn(dataset_fn, batch_size, tilings):
    def input_fn():
        dict_features, labels = dataset_fn() \
        						.shuffle(2000) \
        						.batch(batch_size) \
        						.repeat() \
        						.make_one_shot_iterator() \
        						.get_next()
        features_dict = tilings.get_features_tiles(dict_features)

        return features_dict, labels

    return input_fn

def model_fn(features, labels, mode, params):
    data_in = tf.feature_column.input_layer(features, params['feature_columns'])
    logits  = tf.layers.dense(data_in, units=params['num_classes'], activation=tf.nn.sigmoid)
    loss    = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op  = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(mode, loss=loss)


def main():
    MODEL_DIR   = "model_dir"
    num_tilings = 10
    num_buckets = 10
    batch_size  = 32

    # build input and evaluation functions
    train_fn, evaluation_fn = winequality.get_train_eval_datasets_fn(WINE_EQUALITY_FILE)
    feature_range = winequality.get_feature_range()
    # ---
    tile_strategy_boundaries = TileStrategy(feature_range).uniform(num_buckets)
    tilings = Tilings(tile_strategy_boundaries, num_tilings)

    # ---
    input_fn_train = get_input_fn(train_fn, batch_size, tilings)
    input_fn_eval  = get_input_fn(evaluation_fn, batch_size, tilings)

    # build model function and its necessary params
    tiled_feature_column_list = TiledFeatureColumns(tilings).get_list()
    params = {
        'feature_columns': tiled_feature_column_list,
        'hidden_units': None,
        'num_classes': winequality.get_n_classes()
    }

    # Final training and evaluation. call tensorboard separately to see how loss function evolves
    estimator  = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=MODEL_DIR)
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn_train, max_steps=40000)
    eval_spec  = tf.estimator.EvalSpec(input_fn=input_fn_eval, steps=100, start_delay_secs=0, throttle_secs=30)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    main()
