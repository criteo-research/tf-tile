Tile-Coding Package

pip install -r requirements


Here, we provide a template on how to use the tile-coding for your real-valued input data and also we have provide an example, wine_quality_example.py for empirical
test on UCI winequality dataset.

template:

1. setting the model variables:
a. num_buckets
b. numTilings

2. we have provide a custom estimator model functions,  model_func.py code that can be used for logistic regression or DNNs

3. import tiled_feature_columns and then write


```
import tiled_feature_columns 

tiled_feature_column_list = tiled_feature_columns.get_tiled_feature_columns(numTilings,num_buckets,winequality.FEATURES)
```
Custom Estimator Model function for logistic regression or DNNs is provided:

```
def model_fn(features,labels,mode, params):
  loss = ...
  return tf.estimator.EstimatorSpec(mode, loss=loss)
```

where

```
#parameter input to model_fn used in estimator class
params={
        'feature_columns': tiled_feature_column_list,
        'hidden_units': None,
        'n_classes': winequality.get_n_classes(),
        }
```

Then using estimators as follows should allow to run your experiment and then you should be able to observe the loss function through tensorboard as shown bellow

```
estimator = tf.estimator.Estimator(model_fn=example_model_fn, params=params, model_dir=MODEL_DIR)
train_spec= tf.estimator.TrainSpec(input_fn=lambda: input_func.train_input_fn(batch_size,num_buckets,numTilings),max_steps=max_steps)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_func.eval_input_fn(batch_size,num_buckets,numTilings),steps=steps,start_delay_secs=0,throttle_secs=30)
tf.estimator.train_and_evaluate(estimator,train_spec, eval_spec)
```
