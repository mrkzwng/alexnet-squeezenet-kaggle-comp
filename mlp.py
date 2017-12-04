import tensorflow as tf 


def dense_layer(inputs, input_dim, n_units, iteration, mode):
    '''
    make dense layers with batch normalization and dropout
    '''
    W = tf.get_variable(name='W_'+str(iteration),
                        shape=[input_dim, n_units],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(name='b_'+str(iteration),
                    initial_value=tf.random_uniform(shape=[n_units, ],
                                                    maxval=0.1))
    activ = tf.nn.elu(name='activ_'+str(iteration),
                      features=tf.matmul(inputs, W) + b)
    dropout = tf.layers.dropout(name='dropout_'+str(iteration),
                                inputs=activ,
                                rate=0.5,
                                training=(mode==tf.estimator.ModeKeys.TRAIN))
    norm = tf.layers.batch_normalization(
                    name='norm_'+str(iteration),
                    inputs=dropout,
                    axis=1,
                    training=(mode==tf.estimator.ModeKeys.TRAIN))

    return(norm)


def mlp_function(features, labels, mode, params):
    '''
    assumes features are N x 75 x 75 x 2
    2 channels: band 1 and band 2
    '''
    n_layers = params['n_layers']
    layer_size = params['layer_size']
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

    hidden_output = norm
    for layer_id in xrange(n_layers):
        input_dim = 75*75*2 if layer_id == 0 else layer_size
        hidden_output = dense_layer(inputs=hidden_output,
                                    input_dim=input_dim,
                                    n_units=layer_size,
                                    iteration=layer_id,
                                    mode=mode)

    W_last = tf.get_variable(name='W_last',
                         shape=[layer_size, 2],
                         initializer=tf.contrib.layers.xavier_initializer())
    b_last = tf.get_variable(name='b_last',
                         shape=[2, ],
                         initializer=tf.contrib.layers.xavier_initializer())
    logits = tf.matmul(name='logits', a=hidden_output, b=W_last) + b_last
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
