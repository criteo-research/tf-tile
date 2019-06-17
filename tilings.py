from typing import Dict,List
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
import bucketize_strategy

def tiling_buckets(boundaries,numTilings):
    list_boundaries = []
    each_bucket_resolution = [(float)(boundaries[i+1]-boundaries[i])/numTilings for i in range(len(boundaries)-1)]

    for i in range(numTilings):
        shift_val = []
        for j in range(len(each_bucket_resolution)):
            shift_val.append(i*each_bucket_resolution[j])

        shift_val.append(0)
        
        list_boundaries.append(list(np.array(boundaries)+np.array(shift_val)))

    return list_boundaries


def get_binarized_per_feature(input_data,list_boundaries:List[List[float]]):
    all_buckets = []
    input_tensor = tf.cast(input_data, tf.float64)
    for i,a_boundaries in enumerate(list_boundaries):
        bucketized_tensor = math_ops.bucketize(input_tensor, a_boundaries)
        bucketized_tensor = tf.reshape(bucketized_tensor, (-1, 1))
        bucketized_tensor = tf.math.add(bucketized_tensor,i*(len(a_boundaries)-1)) #added this 
        all_buckets.append(bucketized_tensor)
    return tf.concat(all_buckets, axis=1)


def get_all_sparse_tilings(features,feature_name_range,n_buckets,numTilings):
	features_sparse_tilings = dict()
	for feature_name in feature_name_range.keys(): 
		bucketized_feature_var = bucketize_strategy.uniform_bucketize(feature_name_range[feature_name],n_buckets)
		list_boundaries = tiling_buckets(bucketized_feature_var,numTilings)
		binary_features = get_binarized_per_feature(features[feature_name],list_boundaries)
		features_sparse_tilings[feature_name]=binary_features

	return features_sparse_tilings


