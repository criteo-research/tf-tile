# Here we provide the key functions for tile-coding. To avoid huge dimensionality expansion, we have tiled
# per feature variable, but using feature-column cross functionality a pair of feature-variables
# also can be tiled, and also higher orders.

from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops


class Tilings(object):
    def __init__(self, tile_strategy_boundaries, num_tilings):
        self.num_tilings = num_tilings
        self.tile_strategy_boundaries = tile_strategy_boundaries

    def _get_stack_tiling_boundaries(self, boundaries) -> List[List[float]]:
        boundaries = np.array(boundaries)
        each_bucket_resolution = np.array(
            [float(boundaries[i + 1] - boundaries[i]) / self.num_tilings for i in range(len(boundaries) - 1)] + [0])
        return [list(boundaries + i * each_bucket_resolution) for i in range(self.num_tilings)]

    @staticmethod
    def _get_tiles(input_data, list_boundaries: List[List[float]]):
        all_tiles = []
        input_tensor = tf.cast(input_data, tf.float64)
        for i, boundaries in enumerate(list_boundaries):
            bucketized_tensor = math_ops.bucketize(input_tensor, boundaries)
            bucketized_tensor = tf.reshape(bucketized_tensor, (-1, 1))
            bucketized_tensor = tf.math.add(bucketized_tensor, i * (len(boundaries) - 1)) 
            all_tiles.append(bucketized_tensor)
        return tf.concat(all_tiles, axis=1)

    def get_features_tiles(self, features):
        features_tiles = dict()
        for feature_name, boundaries in self.tile_strategy_boundaries.items():
            list_boundaries = self._get_stack_tiling_boundaries(boundaries)
            features_tiles[feature_name] = Tilings._get_tiles(features[feature_name], list_boundaries)

        return features_tiles
