'''
Here we provide the key functions for tile-coding. To avoid huge dimensionality expansion, we have tiled
per feature variable, but using feature-column cross functionality a pair of feature-variables
also can be tiled, and also higher orders.
'''
from typing import Dict, List
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops


class Tilings(object):
    def __init__(self, tile_strategy_boundaries, num_tilings):
        self.num_tilings = num_tilings
        self.tile_strategy_boundaries = tile_strategy_boundaries

    def get_stack_tiling_boundaries(self, boundaries):
        list_boundaries = []

        each_bucket_resolution = [(float)(boundaries[i + 1] - boundaries[i]) / self.num_tilings for i in
                                  range(len(boundaries) - 1)]

        for i in range(self.num_tilings):
            shift_val = []
            for j in range(len(each_bucket_resolution)):
                shift_val.append(i * each_bucket_resolution[j])

            shift_val.append(0)

            list_boundaries.append(list(np.array(boundaries) + np.array(shift_val)))

        return list_boundaries

    def get_tiles(self, input_data, list_boundaries: List[List[float]]):
        all_tiles = []
        input_tensor = tf.cast(input_data, tf.float64)
        for i, boundaries in enumerate(list_boundaries):
            bucketized_tensor = math_ops.bucketize(input_tensor, boundaries)
            bucketized_tensor = tf.reshape(bucketized_tensor, (-1, 1))
            bucketized_tensor = tf.math.add(bucketized_tensor, i * (len(boundaries) - 1))  # added this
            all_tiles.append(bucketized_tensor)
        return tf.concat(all_tiles, axis=1)

    def get_features_tiles(self, features):
        features_tiles = dict()
        for feature_name, boundaries in self.tile_strategy_boundaries.items():
            list_boundaries = self.get_stack_tiling_boundaries(boundaries)
            features_tiles[feature_name] = self.get_tiles(features[feature_name], list_boundaries)

        return features_tiles
