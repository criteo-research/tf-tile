import pytest
import tensorflow as tf
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),os.pardir))

from tf_tile.tile_strategy import TileStrategy



def test_tile_strategy():
	sess=tf.Session()
	num_buckets = 4
	num_tilings = 3
	feature_name = 'x'
	feature_range ={feature_name: [1,5]}
	dataset =tf.data.Dataset.from_tensor_slices(tf.constant([[1],[2],[3],[4],[5]])) #-> Tuple[Callable[[], tf.data.Dataset], Callable[[], tf.data.Dataset]]:
	dataset = dataset.make_one_shot_iterator().get_next()
	data_dict ={feature_name:dataset}
	tile_strategy_boundaries = TileStrategy(feature_range).uniform(num_buckets)
	assert np.array_equal(tile_strategy_boundaries['x'],range(1,6)) 





