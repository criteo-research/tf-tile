'''
TEST CASE: TO BE COMPLETED
'''

import pytest
import tensorflow as tf
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),os.pardir))

from tf_tile.tiled_feature_columns import TiledFeatureColumns
from tf_tile.tile_strategy import TileStrategy
from tf_tile.tilings import Tilings

def test_tilings():
	sess=tf.Session()
	num_buckets = 4
	num_tilings = 3
	feature_name = 'x'
	feature_range ={feature_name: [1,5]}
	dataset =tf.data.Dataset.from_tensor_slices(tf.constant([[0.9],[1.4]])) #-> Tuple[Callable[[], tf.data.Dataset], Callable[[], tf.data.Dataset]]:
	dataset = dataset.make_one_shot_iterator().get_next()
	data_dict ={feature_name:dataset}
	tile_strategy_boundaries = TileStrategy(feature_range).uniform(num_buckets)
	tilings = Tilings(tile_strategy_boundaries, num_tilings)
	tile_buckets = tilings.get_features_tiles(data_dict)
	bkts_1 = sess.run([tile_buckets['x']])
	bkts_2 = sess.run([tile_buckets['x']])
	
	assert np.array_equal(bkts_1[0][0],[0, 4, 8]) and np.array_equal(bkts_2[0][0],[1, 5, 8])


# def test_tf_tile():
#     sess = tf.Session()

#     num_buckets = 4
#     num_tilings = 3
#     feature_name = 'x'
#     feature_range = {feature_name: [1, 5]}

#     dataset = tf.data.Dataset.from_tensor_slices(tf.constant(
#         [[1], [2], [3], [4], [5]]))  # -> Tuple[Callable[[], tf.data.Dataset], Callable[[], tf.data.Dataset]]:
#     dataset = dataset.make_one_shot_iterator().get_next()
#     data_dict = {feature_name: dataset}

#     tile_strategy_boundaries = TileStrategy(feature_range).uniform(num_buckets)
#     print("tile boundaries are:", tile_strategy_boundaries)
#     tilings = Tilings(tile_strategy_boundaries, num_tilings)

#     tile_column_list = TiledFeatureColumns(tilings).get_list()

#     print(sess.run([data_dict[feature_name]]))
#     data_in = tf.feature_column.input_layer(data_dict, tile_column_list)

#     print(sess.run([data_in]))




