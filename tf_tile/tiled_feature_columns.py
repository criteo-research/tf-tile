import tensorflow as tf


class TiledFeatureColumns:

    def __init__(self, tilings):
        self.num_tilings = tilings.num_tilings
        self.tile_strategy_boundaries = tilings.tile_strategy_boundaries

    def get_list(self):
        tiled_feature_column_list = []

        for key in self.tile_strategy_boundaries:
            tiled_feature_column_list.append(
                tf.feature_column.indicator_column(
                    tf.feature_column.categorical_column_with_identity(
                        key,
                        num_buckets=self.num_tilings * len(self.tile_strategy_boundaries[key]) 
                    )
                )
                )
        return tiled_feature_column_list


# TO DO: for more efficient computation
def get_tiled_feature_columns_sparse(num_tilings, num_buckets, feature_var_list):
    tiled_features_sparse = []
    for key in feature_var_list:
        tiled_features_sparse.append(
            tf.feature_column.categorical_column_with_identity(
                key,
                num_buckets=num_tilings * num_buckets + 2
            ))

    return tiled_features_sparse
