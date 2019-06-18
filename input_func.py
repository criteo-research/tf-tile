import tensorflow as tf

def get_input_fn(dataset, batch_size, tilings):
	
	def input_fn():
		dict_features,labels = dataset.shuffle(2000).batch(batch_size).repeat().make_one_shot_iterator().get_next()
		features_dict = tilings.get_features_tiles(dict_features)
	
		return features_dict,labels

	return input_fn



