'''
Using tile-coding technique as an efficient way of sparse-coding for real-valued data
Here, winequality dataset has been used as an example 
'''
import tensorflow as tf
import winequality
import input_func
import model_func
import tiled_feature_columns
from tensorflow.python.platform import tf_logging as logging
from tile_strategy import TileStrategy
from tiled_feature_columns import TiledFeatureColumns
from tilings import Tilings
tf.logging.set_verbosity(tf.logging.INFO)

MODEL_DIR        =  "model_dir"
num_tilings      =  10
num_buckets      =  10
batch_size       =  32

#build input and evaluation functions
train, evaluation     = winequality.get_train_eval_datasets(winequality.FILE_NAME)
feature_range    = winequality.get_feature_range()
#---
tile_strategy_boundaries = TileStrategy(feature_range).uniform(num_buckets)
tilings = Tilings(tile_strategy_boundaries,num_tilings)


#---
input_fn_train = input_func.get_input_fn(train, batch_size,tilings)
input_fn_eval = input_func.get_input_fn(evaluation, batch_size,tilings)

# build model function and its necessary params
example_model_fn          = model_func.model_fn
tiled_feature_column_list = TiledFeatureColumns(tilings).get_list()
params                    = {
                            'feature_columns': tiled_feature_column_list,
                            'hidden_units': None,
                            'num_classes': winequality.get_n_classes()
                            }

# #Final training and evaluation. call tensorboard separately to see how loss function evolves
estimator = tf.estimator.Estimator(model_fn=example_model_fn, params=params, model_dir=MODEL_DIR)
train_spec= tf.estimator.TrainSpec(input_fn=input_fn_train,max_steps=40000)
eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_eval ,steps=100,start_delay_secs=0,throttle_secs=30)
tf.estimator.train_and_evaluate(estimator,train_spec, eval_spec)
