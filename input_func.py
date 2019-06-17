import tensorflow as tf
import tilings


def get_input_fn(dataset, feature_range, batch_size, num_buckets,num_tilings):
	def input_fn():
		dict_features,labels = dataset.shuffle(2000).batch(batch_size).repeat().make_one_shot_iterator().get_next()
		features_dict = tilings.get_all_sparse_tilings(dict_features,feature_range,num_buckets,num_tilings)
	
		return features_dict,labels

	return input_fn



