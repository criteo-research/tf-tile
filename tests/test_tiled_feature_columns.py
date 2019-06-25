import pytest
import tensorflow as tf
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),os.pardir))

from tf_tile.tiled_feature_columns import TiledFeatureColumns
from tf_tile.tile_strategy import TileStrategy
from tf_tile.tilings import Tilings

def test_tiled_feature_columns():
	sess=tf.Session()
	num_buckets = 4
	num_tilings = 3
	feature_name = 'x'
	feature_range ={feature_name: [1,5]}
	dataset =tf.data.Dataset.from_tensor_slices(tf.constant([[0.9],[1.4]]))
	dataset = dataset.make_one_shot_iterator().get_next()
	data_dict ={feature_name:dataset}
	tile_strategy_boundaries = TileStrategy(feature_range).uniform(num_buckets)
	tilings = Tilings(tile_strategy_boundaries, num_tilings)
	data_dict_tiled = tilings.get_features_tiles(data_dict)
	tile_column_list = TiledFeatureColumns(tilings).get_list()

	data_in = tf.feature_column.input_layer(data_dict_tiled, tile_column_list)

	data_1 = sess.run([data_in])
	expected_arr1=  [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
	data_2 = sess.run([data_in])
	expected_arr2 = [0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
	
	
	assert  np.array_equal(data_1[0][0],expected_arr1) and np.array_equal(data_2[0][0],expected_arr2)