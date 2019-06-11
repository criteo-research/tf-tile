import math
import logging
import tensorflow as tf
from .features import features_dnn
from .vocabulary import load_vocabularies


def dnn_model_fn(features, labels, mode, params):
   
    with tf.name_scope("input_columns"):
    
        net = tf.feature_column.input_layer(features, params['features_columns'])

    # Build the hidden layers, sized according to the 'hidden_units' param.
    for layer_index, units in enumerate(params['hidden_units']):
        with tf.name_scope(f"layer_{layer_index}"):
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    with tf.name_scope("output"):
        # Compute logits
        predictions = tf.layers.dense(net, 1, activation=tf.nn.sigmoid)
        predicted_classes = tf.rint(predictions)

    ################
    # Predict mode #
    ################

    logging.info("Keys in features are {}".format(features.keys()))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'scores': predictions
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.name_scope("optimizer"):
        loss = tf.losses.log_loss(labels=labels, predictions=predictions)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # with tf.name_scope("metrics"):
    #     accuracy = tf.metrics.accuracy(labels=labels,
    #                                    predictions=predicted_classes,
    #                                    name='acc_op')

    #     auc = tf.metrics.auc(labels=labels, predictions=predictions, name='auc_op')

    #     false_positives = tf.metrics.false_positives(labels=labels, predictions=predicted_classes)
    #     false_negatives = tf.metrics.false_negatives(labels=labels, predictions=predicted_classes)
    #     true_positives = tf.metrics.true_positives(labels=labels, predictions=predicted_classes)
    #     true_negatives = tf.metrics.true_negatives(labels=labels, predictions=predicted_classes)

    ##############
    # Evaluation #
    ##############

    # metrics = {
    #     'auc': auc,
    #     'accuracy': accuracy,
    #     'true_positives': true_positives,
    #     'true_negatives': true_negatives,
    #     'false_positives': false_positives,
    #     'false_negatives': false_negatives
    # }

    # if mode == tf.estimator.ModeKeys.EVAL:
    #     return tf.estimator.EstimatorSpec(
    #         mode, loss=loss, eval_metric_ops=metrics)

    #########
    # Train #
    #########

    #assert mode == tf.estimator.ModeKeys.TRAIN

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


