import tensorflow as tf
import tilings


def get_input_fn(dataset, feature_range, batch_size, num_buckets,num_tilings):
	def input_fn():
		dict_features,labels = dataset.shuffle(2000).batch(batch_size).repeat().make_one_shot_iterator().get_next()
		features_dict = tilings.get_all_sparse_tilings(dict_features,feature_range,num_buckets,num_tilings)
	
		return features_dict,labels

	return input_fn



# def train_input_fn(train,feature_range,batch_size,n_buckets,numTilings):

# 	#shuffle size is hardcoded and can be defined in config

# 	dict_features,labels = train.shuffle(2000).batch(batch_size).repeat().make_one_shot_iterator().get_next()
# 	features_dict = tilings.get_all_sparse_tilings(dict_features,feature_range,n_buckets,numTilings)
	
# 	return features_dict,labels

# def eval_input_fn(test,feature_range,batch_size,n_buckets,numTilings):

# 	dict_features,labels = test.shuffle(2000).batch(batch_size).repeat().make_one_shot_iterator().get_next()
# 	features_dict = tilings.get_all_sparse_tilings(dict_features,feature_range,n_buckets,numTilings)
	
# 	return features_dict,labels


