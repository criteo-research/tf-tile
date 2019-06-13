from typing import Dict,List
import winequality
import tensorflow as tf
import numpy as np
import bucketize_strategy
import tilings
import data_stats
import winequality


train, test = winequality.get_train_eval_datasets(winequality.FILE_NAME)
feature_range = data_stats.get_feature_range()


	
#input_data = [dict_features['fixed_acidity']]


def train_input_fn(batch_size,n_buckets,numTilings):

	dict_features,labels = train.shuffle(2000).batch(batch_size).repeat().make_one_shot_iterator().get_next()
	features_dict = tilings.get_all_sparse_tilings(dict_features,feature_range,n_buckets,numTilings)
	
	return features_dict,labels

def eval_input_fn(batch_size,n_buckets,numTilings):

	dict_features,labels = test.shuffle(2000).batch(batch_size).repeat().make_one_shot_iterator().get_next()
	features_dict = tilings.get_all_sparse_tilings(dict_features,feature_range,n_buckets,numTilings)
	
	return features_dict,labels


