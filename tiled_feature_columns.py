import tensorflow as tf

def get_tiled_feature_columns(numTilings,num_buckets, feature_var_list):

	tiled_feature_column_list = []
	for key in feature_var_list:
		tiled_feature_column_list.append(
							#indicator_column
	                        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
	                        key,
	                        num_buckets=numTilings*num_buckets+2
	                        ))
	                        )
	return tiled_feature_column_list

#TO DO: for more efficient computation
def get_tiled_feature_columns_sparse(numTilings,num_buckets, feature_var_list):

	tiled_features_sparse = []
	for key in feature_var_list:
		tiled_features_sparse.append(
	                        tf.feature_column.categorical_column_with_identity(
	                        key,
	                        num_buckets=numTilings*num_buckets+2
	                        ))
	                        
	return tiled_features_sparse