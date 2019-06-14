import tensorflow as tf

def model_fn(features,labels,mode, params):

    data_in = tf.feature_column.input_layer(features, params['feature_columns'])
    logits = tf.layers.dense(data_in, units=params['n_classes'], activation=tf.nn.sigmoid)
    loss =  tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    eval_metric = {'loss': loss}
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

