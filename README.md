Tile-Coding in TensorFlow ---a sparse-coding tool for generating features from real-valued data

```
pip install -r requirements
```

Here, we show how to use tile-coding in your TF code through an example. Here, we use winequality UCI as an example.

The real-valued feature-names are as follows:

```
FEATURES = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide', 'total_sulfur_dioxide','density','pH','sulphates','alcohol']
```

then set desirable values for num_buckets (e.g. 10) and num_tilings (e.g. 10) and call:

```
tiled_feature_column_list = tiled_feature_columns.get_tiled_feature_columns(num_tilings,num_buckets,FEATURES)
```

Now the tiled_feature_column_list can be passed into params dictionary that will be used in TF estimator class. For example, for the case of calssification we use
logisitic regression using custom estimators with num_classes, we can set:

```
params={
        'feature_columns': tiled_feature_column_list,
        'hidden_units': None,
        'n_classes': num_classes
        }

estimator = tf.estimator.Estimator(model_fn=example_model_fn, params=params, model_dir=MODEL_DIR)
```

Now in the input_fn (created for both train and eval) we tile the real-value data. For example,

```
train, test = winequality.get_train_eval_datasets(winequality.FILE_NAME)
feature_range = winequality.get_feature_range()
def train_input_fn(train,feature_range,batch_size,n_buckets,numTilings):

	#shuffle size is hardcode and can be defined in config

	dict_features,labels = train.shuffle(2000).batch(batch_size).repeat().make_one_shot_iterator().get_next()
	features_dict = tilings.get_all_sparse_tilings(dict_features,feature_range,n_buckets,numTilings)
	
	return features_dict,labels
	
train_spec= tf.estimator.TrainSpec(input_fn=lambda: input_func.train_input_fn(batch_size,num_buckets,numTilings),max_steps=max_steps)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_func.eval_input_fn(batch_size,num_buckets,numTilings),steps=steps,start_delay_secs=0,throttle_secs=30)
tf.estimator.train_and_evaluate(estimator,train_spec, eval_spec)

```

