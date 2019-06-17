import typing

import tensorflow as tf

FILE_NAME="winequality-red.csv" #"cleaned-winequality-red.csv"
LABEL = "category" #"quality"

FEATURES = [
'fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide',
'total_sulfur_dioxide','density','pH','sulphates','alcohol'
]

feature_range_list = [[4.6,15.9],[0.12,1.33],[0,1.0],[0.9,9],[0.012,0.27],[1.0,68.0],[6.0,165.0],[0.99007,1.00369],[2.74,4.01],[0.33,1.36],[8.4,14.9]]

def get_feature_range():
    feature_range = dict()
    for i,k in enumerate(FEATURES):
        feature_range[k]= feature_range_list[i]
    return feature_range



def get_train_eval_datasets(
    path: str,
    train_fraction: float = 0.7
) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
    def split_label(*row):

        return dict(zip(FEATURES, row)), row[-1]

    def in_training_set(*row):
        num_buckets = 1000
        key = tf.string_join(list(map(tf.as_string, row)))
        bucket_id = tf.string_to_hash_bucket_fast(key, num_buckets)
        return bucket_id < int(train_fraction * num_buckets)

    def in_test_set(*row):
        return ~in_training_set(*row)

    data = tf.data.experimental.CsvDataset(
        path,
        [tf.float32] * len(FEATURES) +[tf.int32],
        header=True,
        field_delim=";")

    train = data.filter(in_training_set).map(split_label).cache()
    test = data.filter(in_test_set).map(split_label).cache()
    return train, test


def get_feature_columns():
    return [tf.feature_column.numeric_column(name) for name in FEATURES]


def get_n_classes():
    return 9 
