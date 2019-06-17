import tensorflow as tf

def model_fn(features,labels,mode, params):

    data_in = tf.feature_column.input_layer(features, params['feature_columns'])
    logits = tf.layers.dense(data_in, units=params['num_classes'], activation=tf.nn.sigmoid)
    loss =  tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    eval_metric = {'loss': loss}
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(mode, loss=loss)


#-----------------------------------------------------------------
# TO DO: an sparse implementation for a more efficient computation
#-----------------------------------------------------------------

def fully_connect(features,weight_size,n_classes):

    weights = tf.get_variable("weights", weight_size, initializer=tf.initializers.zeros())
    biases = tf.get_variable("biases", n_classes, initializer=tf.initializers.zeros())
    return tf.nn.embedding_lookup_sparse(params=weights, sp_ids=features,sp_weights=None , combiner="sum") + biases

def model_fn_sparse(features,labels,mode, params):



    logits = fully_connect(features,params['weight_size'],params['n_classes'])

    loss = tf.losses.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    eval_metric = {'loss': loss}
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(mode, loss=loss)
