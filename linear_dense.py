import tensorflow as tf 


def linear_dense_function(features, labels, mode, params):
    '''
    assumes features are N x 75 x 75 x 2
    2 channels: band 1 and band 2
    '''
    # ADAM hyper parameters
    learning_rate = params['learning_rate']
    moment_decay = params['moment_decay']
    moment2_decay = params['moment2_decay']

    '''
    single-layer preceptron
    '''
    # data
    X = tf.cast(features['x'], dtype=tf.float32)
    '''
    fully-connected layer
    [N, 75, 75, 2] -> [N, 2]
    '''
    X = tf.reshape(X, [-1, 75 * 75 * 2])
    norm = tf.layers.batch_normalization(
                    name='norm',
                    inputs=X,
                    axis=1,
                    training=(mode==tf.estimator.ModeKeys.TRAIN))
    W1 = tf.get_variable(name='W1',
                         shape=[75 * 75 * 2, 2],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(name='b1',
                         shape=[2, ],
                         initializer=tf.contrib.layers.xavier_initializer())
    logits = tf.matmul(name='logits', a=norm, b=W1) + b1
    predictions = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        one_hot_labels = tf.one_hot(indices=labels, depth=2)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=moment_decay,
                                      beta2=moment2_decay)
        norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(norm_update_ops):
            update_op = adam.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return(tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=update_op))

    elif mode == tf.estimator.ModeKeys.PREDICT:
        return(tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions))

    elif mode == tf.estimator.ModeKeys.EVAL:
        one_hot_labels = tf.one_hot(indices=labels, depth=2)
        loss = tf.losses.log_loss(one_hot_labels, predictions)
        predicted_labels = tf.argmax(input=predictions, axis=1)
        metrics = {'false_negatives': tf.metrics.false_negatives(
                                    labels=labels,
                                    predictions=predicted_labels),
                   'true_positives': tf.metrics.true_positives(
                                    labels=labels,
                                    predictions=predicted_labels),
                   'false_positives': tf.metrics.false_positives(
                                    labels=labels,
                                    predictions=predicted_labels),
                   'precision': tf.metrics.precision(
                                    labels=labels,
                                    predictions=predicted_labels),
                   'recall': tf.metrics.recall(
                                    labels=labels,
                                    predictions=predicted_labels),
                   'AUC': tf.metrics.auc(
                                    labels=labels,
                                    predictions=predicted_labels),
                   'mean_per_class_acc': tf.metrics.mean_per_class_accuracy(
                                    labels=labels,
                                    predictions=predicted_labels,
                                    num_classes=2),
                   'accuracy': tf.metrics.accuracy(
                                    labels=labels,
                                    predictions=predicted_labels)}
        return(tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=metrics))
