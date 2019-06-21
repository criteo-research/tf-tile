'''
TEST CASE: TO BE COMPLETED
'''

import pytest
from typing import Dict, List
import tensorflow as tf
import sys
sys.path.insert(0, "../")
from tf_tile.tiled_feature_columns import TiledFeatureColumns
from tf_tile.tile_strategy import TileStrategy
from tf_tile.tilings import Tilings


sess=tf.Session()

num_buckets = 4
num_tilings = 3
feature_name = 'x'
feature_range ={feature_name: [1,5]}

dataset =tf.data.Dataset.from_tensor_slices(tf.constant([[1],[2],[3],[4],[5]])) #-> Tuple[Callable[[], tf.data.Dataset], Callable[[], tf.data.Dataset]]:
dataset = dataset.make_one_shot_iterator().get_next()
data_dict ={feature_name:dataset}

tile_strategy_boundaries = TileStrategy(feature_range).uniform(num_buckets)
print("tile boundaries are:", tile_strategy_boundaries)
tilings = Tilings(tile_strategy_boundaries, num_tilings)

tile_column_list = TiledFeatureColumns(tilings).get_list() 

print(sess.run([data_dict[feature_name]]))
data_in = tf.feature_column.input_layer(data_dict, tile_column_list)


print(sess.run([data_in]))




