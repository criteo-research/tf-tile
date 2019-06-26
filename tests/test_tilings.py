'''
Test case for tilings.py. This is the most crucial test case where it shows which tile indices will be activated
given the num_tilings and tile-strategy
'''

import numpy as np
import tensorflow as tf

from tf_tile.tile_strategy import TileStrategy
from tf_tile.tilings import Tilings


def test_tilings():
    sess = tf.Session()
    num_buckets = 4
    num_tilings = 3
    feature_name = 'x'
    feature_range = {feature_name: [1, 5]}
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant([[0.9], [1.4]]))
    dataset = dataset.make_one_shot_iterator().get_next()
    data_dict = {feature_name: dataset}
    tile_strategy_boundaries = TileStrategy(feature_range).uniform(num_buckets)
    tilings = Tilings(tile_strategy_boundaries, num_tilings)
    tile_buckets = tilings.get_features_tiles(data_dict)
    bkts_1 = sess.run([tile_buckets['x']])
    bkts_2 = sess.run([tile_buckets['x']])

    assert np.array_equal(bkts_1[0][0], [0, 4, 8]) and np.array_equal(bkts_2[0][0], [1, 5, 8])
