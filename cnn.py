import tensorflow as tf
import numpy as np

def conv_reduce_mod(inputs, input_dim, output_dim, drop_rate, iteration, mode):
    '''
    factory conv layer with batch normalization and maxpool

    reduces W x H dimensions via max pooling
    '''
    W = tf.get_variable(name='W_reduce'+str(iteration),
                        shape=[2, 2, input_dim, output_dim],
                        initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(name='conv_reduce'+str(iteration),
                        input=inputs,
                        filter=W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    maxpool = tf.nn.max_pool(name='max_reduce'+str(iteration),
                             value=conv,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
    activ = tf.nn.elu(name='activ_reduce'+str(iteration),
                      features=maxpool)
    dropout = tf.layers.dropout(name='drop_reduce_'+str(iteration),
                                inputs=activ,
                                rate=drop_rate,
                                training=(mode==tf.estimator.ModeKeys.TRAIN))
    norm = tf.layers.batch_normalization(
                        name='norm_reduce'+str(iteration),
                        inputs=dropout,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))
    return(norm)


def conv_mod(inputs, input_dim, output_dim, drop_rate, iteration, mode):
    '''
    factory conv module with batch normalization and dropout
    '''
    W = tf.get_variable(name='W_'+str(iteration),
                        shape=[3, 3, input_dim, output_dim],
                        initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(name='conv_'+str(iteration),
                        input=inputs,
                        filter=W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    activ = tf.nn.elu(name='activ_'+str(iteration),
                      features=conv)
    dropout = tf.layers.dropout(name='dropout_'+str(iteration),
                                inputs=activ,
                                rate=drop_rate,
                                training=(mode==tf.estimator.ModeKeys.TRAIN))
    norm = tf.layers.batch_normalization(
                        name='norm_'+str(iteration),
                        inputs=dropout,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))
    return(norm)


def cnn_function(features, labels, params, mode):
    '''
    custom convnet
    '''
    learning_rate = params['learning_rate']
    moment1 = params['moment1']
    moment2 = params['moment2']

    conv_layers = params['conv_layers']
    reduce_layers = params['reduce_layers']
    conv_chs = params['conv_chs']
    drop_rate = params['drop_rate']

    X = tf.cast(features['x'], dtype=tf.float32)

    conv1_in = X
    output_dim = 0
    for layer_id in xrange(conv_layers):
        input_dim = input_dim + output_dim if layer_id > 0 else 2
        output_dim = output_dim * 2 if layer_id > 0 else conv_chs
        conv1_out = conv_mod(inputs=conv1_in,
                             input_dim=input_dim,
                             output_dim=output_dim,
                             drop_rate=drop_rate,
                             iteration=layer_id,
                             mode=mode)
        conv1_in = tf.concat([conv1_in, conv1_out], axis=3)

    reduce1_out = conv1_in
    input_dim = input_dim + output_dim
    height_width = 75
    for layer_id in xrange(reduce_layers):
        reduce1_out = conv_reduce_mod(inputs=reduce1_out,
                                      input_dim=input_dim,
                                      output_dim=input_dim * 2,
                                      drop_rate=drop_rate,
                                      iteration=layer_id,
                                      mode=mode)
        height_width = int(np.ceil(height_width / 2.0))
        input_dim = input_dim * 2

    reduce1_out = tf.reshape(reduce1_out, 
                             shape=[-1, input_dim * height_width ** 2])
    W_last = tf.get_variable(name='W_last',
                             shape=[input_dim * height_width ** 2, 2],
                             initializer=tf.contrib.layers.xavier_initializer())
    b_last = tf.Variable(name='b_last',
                         initial_value=tf.random_uniform(
                                shape=[2, ],
                                maxval=0.1))
    logits_sm = tf.matmul(reduce1_out, W_last) + b_last
    softmax = tf.nn.softmax(name='softmax', logits=logits_sm)


    if tf.estimator.ModeKeys.TRAIN == mode:
        y = tf.one_hot(indices=labels, depth=2)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, 
                                               logits=logits_sm)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=moment1,
                                           beta2=moment2)
        norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(norm_update_ops):
            minimize_op = optimizer.minimize(loss=loss,
                                             global_step=tf.train.get_global_step())
        return(tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=minimize_op))

    if tf.estimator.ModeKeys.PREDICT == mode:
        y_hat = tf.argmax(input=softmax, axis=1)
        return(tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=softmax))

    if tf.estimator.ModeKeys.EVAL == mode:
        y = tf.one_hot(indices=labels, depth=2)
        loss = tf.losses.log_loss(labels=y, 
                                  predictions=softmax)
        y_hat = tf.argmax(input=softmax, axis=1)
        metrics = {'false_negatives': tf.metrics.false_negatives(
                                    labels=labels,
                                    predictions=y_hat),
                   'true_positives': tf.metrics.true_positives(
                                    labels=labels,
                                    predictions=y_hat),
                   'false_positives': tf.metrics.false_positives(
                                    labels=labels,
                                    predictions=y_hat),
                   'precision': tf.metrics.precision(
                                    labels=labels,
                                    predictions=y_hat),
                   'recall': tf.metrics.recall(
                                    labels=labels,
                                    predictions=y_hat),
                   'AUC': tf.metrics.auc(
                                    labels=labels,
                                    predictions=y_hat),
                   'mean_per_class_acc': tf.metrics.mean_per_class_accuracy(
                                    labels=labels,
                                    predictions=y_hat,
                                    num_classes=2),
                   'accuracy': tf.metrics.accuracy(
                                    labels=labels,
                                    predictions=y_hat)}
        return(tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=metrics))