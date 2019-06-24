tf-tile
=======
Tile-Coding in TensorFlow: a sparse-coding tool for generating features from real-valued data)

What is Tile-Coding? When it comes to real-valued data discretization (also called bucketization) of data is a common choice to normalize and tranform data to a
a feature vector that is ready to be used in classical machine learning methods such as logistic regression and recently the first layer of 
deep neural networks (DNNs). However, bucketizing the input data has several drawbacks as follows: a) It can have a large prediction error due to the resolution 
of bucket-size. b) The dimensionality of feature-vector, and thus model-size increases if a much lower resolution has been selected for the bucket-size, b)
It lacks generalization as we explain  in the figure bellow.

![alt text](tf_tile_pic.jpg)

Installation
------------

Create virtual environment (for example called tf-tile) for python using conda:
* [Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Steps to run the example
-----------------------

1) set up the environment and dependencies:

```bash
$ conda create -n tf-tile python=3.6
$ conda activate tf-tile
$ cd tf-tile
$ pip install -r requirements.txt
```

2) Download winequality-red.csv from the Wine Quality dataset at UCI ML repository (https://archive.ics.uci.edu/ml/datasets/Wine+Quality).

3) You can now test this out by trying to run one of our examples!
```bash
$ cd examples
$ python winequality_example.py
...
I0624 10:21:46.237035 4663645632 session_manager.py:500] Running local_init_op.
I0624 10:21:46.257230 4663645632 session_manager.py:502] Done running local_init_op.
I0624 10:21:46.908173 4663645632 basic_session_run_hooks.py:606] Saving checkpoints for 0 into model_dir/model.ckpt.
I0624 10:21:48.018400 4663645632 basic_session_run_hooks.py:262] loss = 2.1749778, step = 1
I0624 10:21:48.691524 4663645632 basic_session_run_hooks.py:692] global_step/sec: 147.251
...
```

Winequality example
-------------------

Here, we show how to use tile-coding in your TF code through an example. We use [winequality UCI](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) as an example, where the real-valued feature-names are as follows:

```
FEATURES = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide', 'total_sulfur_dioxide','density','pH','sulphates','alcohol']
```

The first step is to decide how to discretize the real-valued data and choose the number of buckets num_buckets (e.g. 10) and thus boundaries. We assign a class 
to this step called TileStrategy. The usual strategy is uniform but the user can assign different tiles using data statistics (custom tiles).

```
tile_strategy_boundaries = TileStrategy(feature_range).uniform(num_buckets) #or build your own custom tiling strategy, for example using kernel desnsity estimation 
```


Then the second stage will be tiling the data given tile_strategy_boundaries, which is a dictionary holding boundaries for each feature. For tilings we can set
a suitable value for num_tilings (e.g. 10) and  for more information on tilings you may visit: https://confluence.criteois.com/display/~h.maei/Tile-Coding%3A+An+Efficient+Sparse-Coding+Method+for+Real-Valued+Data#link-talk-213056

```
tilings = Tilings(tile_strategy_boundaries,num_tilings)
```

Now following the classical template used in tensorflow for estimator class we compute input functions with tiled features. For example,

```
input_fn_train = get_input_fn(train, batch_size,tilings)
input_fn_eval = ...
```
Once the custom model function is built (again see the example), 

```
tiled_feature_column_list = TiledFeatureColumns(tilings).get_list()
params                    = {
                            'feature_columns': tiled_feature_column_list,
                            'hidden_units': None,
                            'num_classes': winequality.get_n_classes()
                            }
```

and now everything should be ready for training:

```
estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=MODEL_DIR)
train_spec= tf.estimator.TrainSpec(input_fn=input_fn_train,max_steps=40000)
eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_eval ,steps=100,start_delay_secs=0,throttle_secs=30)
tf.estimator.train_and_evaluate(estimator,train_spec, eval_spec)
```

You can see the full example here: [examples/winequality_example.py](examples/winequality_example.py)
