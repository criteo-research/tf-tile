'''
Test the tiling strategy. The test case here is for the case of uniform tiling--uniform bucketization given the
range of input variables and also the number of buckets.
'''

import numpy as np

from tf_tile.tile_strategy import TileStrategy


def test_tile_strategy():
    num_buckets = 4
    feature_name = 'x'
    feature_range = {feature_name: [1, 5]}
    tile_strategy_boundaries = TileStrategy(feature_range).uniform(num_buckets)
    assert np.array_equal(tile_strategy_boundaries['x'], range(1, 6))
