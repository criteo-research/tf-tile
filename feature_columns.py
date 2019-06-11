from typing import Dict,List
from tensorflow.python.ops import math_ops
import winequality
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import tf_logging as logging
import bucketize_strategy
import tilings
import input_func
tf.logging.set_verbosity(tf.logging.INFO)

HDFS_DIR="tmp"
numTilings=10
num_buckets = 20
batch_size = 64
tiled_feature_column_list = []
for key in winequality.FEATURES:

    tiled_feature_column_list.append(
                        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
                        key,
                        num_buckets=numTilings*(num_buckets+1)
                        ))
                        )


params={
        'feature_columns': tiled_feature_column_list,
        'hidden_units': None,
        'n_classes': winequality.get_n_classes()
        }

def example_model_fn(features,labels,mode, params):
    data_in = tf.feature_column.input_layer(features, params['feature_columns'])
    logits = tf.layers.dense(data_in, units=winequality.get_n_classes(), activation=tf.nn.sigmoid)
    loss =  tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    
classifier = tf.estimator.Estimator(model_fn=example_model_fn, params=params)
classifier.train(input_fn=lambda: input_func.train_input_fn(batch_size,num_buckets,numTilings),steps=100000)   

#train_spec= tf.estimator.TrainSpec(train_input_fn, max_steps=1000)
#eval_spec = tf.estimator.EvalSpec(train_input_fn,steps=100,start_delay_secs=0,throttle_secs=30)
#tf.estimator.train_and_evaluate(estimator,train_spec, eval_spec)
# def get_feature_columns():
#     return [tf.feature_column.numeric_column(name) for name in FEATURES]

# def get_feature_columns():
#     return [tiling_feature(name) for name in FEATURES]







# import collections
# def tree(): return collections.defaultdict(tree)

# tiled_features = tree()
# numTilings     = 8

# tiled_features["nbsale_1week"]["boundaries"] =[...]


# # Tiled feature columns => "my_feature_tiled_col"
# ALL.update({(name + "_tiled_col"): CategoricalColumnWithIdentity(
#     feature_id=(name + "_tiled_col"),
#     dependencies=[name + "_tiled"],
#     column_dependencies=[],
#     col_cardinality=len(params[0])*params[2]+1)  #added from params[2]+1 #params[0] is boundaries and params[2] is numTilings
#     for name, params in tiled_features_with_resolution.items()
# })


# class CustomFeature(ABC):
#     def __init__(self, feature_id: str, dtype: tf.DType, dependencies: List[str]):
#         self.feature_id = feature_id
#         self.dtype = dtype
#         self.dependencies = dependencies

#     def build(self, data: Dict) -> Dict:
#         data[self.feature_id] = tf.cast(self._build(data), dtype=self.dtype)
#         return data

#     @abstractmethod
#     def _build(self, data: Dict) -> tf.Tensor:
#         pass

#     def __repr__(self):
#         return f"{self.feature_id} of dtype {self.dtype} depends on {self.dependencies}"


# class TiledFeature(CustomFeature):
#     def __init__(self,
#                  feature_id: str,
#                  dependencies: List[str],
#                  boundaries: List[List[float]]
#                  ):
#         CustomFeature.__init__(self, feature_id, tf.int64, dependencies)
#         self.boundaries = boundaries

#     def _build(self, data: Dict) -> tf.Tensor:
#         a_bucket = []
#         input_tensor = tf.cast(data[self.dependencies[0]], tf.float64)
#         for i,a_boundaries in enumerate(self.boundaries):
#             bucketized_tensor = math_ops.bucketize(input_tensor, a_boundaries)
#             bucketized_tensor = tf.reshape(bucketized_tensor, (-1, 1))
#             bucketized_tensor = tf.math.add(bucketized_tensor,i*(len(a_boundaries)-1)) #added this 
#             a_bucket.append(bucketized_tensor)
#         return tf.concat(a_bucket, axis=1)
#         #return tf.clip_by_value(t,0,1.01)

# class CategoricalColumnWithIdentity(FeatureColumn):
#     def __init__(self, feature_id: str, dependencies: List[str], column_dependencies: List[str],
#                  col_cardinality: int = DEFAULT_LOGROUND_IDENTITY_CARDINALITY):
#         FeatureColumn.__init__(self, feature_id, dependencies, column_dependencies)
#         self.col_cardinality = col_cardinality

#     def build(self, features: List[Union[RawFeature, CustomFeature]]=None,
#               built_column_dependencies: Optional[List[NamedTuple]]=None) -> NamedTuple:
#         return tf.feature_column.categorical_column_with_identity(
#             features[0].feature_id, self.col_cardinality + 1, default_value=self.col_cardinality
#         )

# class TiledFeatureWithResolution(TiledFeature):
#     def __init__(self,
#                  feature_id: str,
#                  dependencies: List[str],
#                  boundaries: List[float],
#                  increments: List[float], #changed
#                  nb_tiles: int
#                  ):
#         all_boundaries = [
#             [boundaries[j]+i*increments[j] for j in range(len(boundaries))]
#             for i in range(nb_tiles)
#         ]
#         #print("all_boundaries:", all_boundaries)
#         TiledFeature.__init__(self, feature_id, dependencies, all_boundaries)

# for key in tiled_features:

#     tiled_features[key]["numTilings"] = numTilings
    
#     boundaries = tiled_features[key]["boundaries"]

#     tiled_features[key]["increments"]= [(float)(boundaries[i+1]-boundaries[i])/numTilings for i in range(len(boundaries)-1)]
#     tiled_features[key]["increments"].append(0)

# tiled_features_with_resolution= {}
# for key in tiled_features:
#     tiled_features_with_resolution[key] = (tiled_features[key]["boundaries"],
#                                             tiled_features[key]["increments"],
#                                             tiled_features[key]["numTilings"])
# #------

# ALL.update({
#     f"{feature_id}_tiled": TiledFeatureWithResolution(
#         feature_id=f"{feature_id}_tiled",
#         dependencies=[feature_id],
#         boundaries=boundaries,
#         increments=increments,
#         nb_tiles=nb_tiles
#     )
#     for feature_id, (boundaries, increments, nb_tiles) in tiled_features_with_resolution.items()
# })

