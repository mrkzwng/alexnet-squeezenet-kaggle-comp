import tensorflow as tf


def alexnet_function(features, labels, mode, params):
    '''
    assumes features are N x 75 x 75 x C
    N = batch size
    C = number of channels
    '''
    learning_rate = params['learning_rate']
    moment1 = params['moment1']
    moment2 = params['moment2']
    in_channels = params['in_channels']
    conv_k_chs = params['conv_k_chs']
    fc1_units = params['fc1_units']
    fc2_units = params['fc2_units']
    dropout_rate = params['dropout_rate']

    is_training = mode == 'train'
    
    X = tf.cast(features['x'], dtype=tf.float32)

    # conv layer 1
    W_1a = tf.get_variable(name='W_1a',
                           shape=[5, 5, in_channels, conv_k_chs[0]],
                           initializer=tf.contrib.layers.xavier_initializer())
    conv_1a = tf.nn.conv2d(name='conv_1a',
                           input=X,
                           filter=W_1a,
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    max_1a = tf.nn.max_pool(name='max_1a',
                        value=conv_1a,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    activ_1a = tf.nn.elu(name='activ_1a',
                         features=max_1a)
    norm_1a = tf.layers.batch_normalization(
                        name='norm_1a',
                        inputs=activ_1a,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    W_1b = tf.get_variable(name='W_1b',
                           shape=[5, 5, in_channels, conv_k_chs[0]],
                           initializer=tf.contrib.layers.xavier_initializer())
    conv_1b = tf.nn.conv2d(name='conv_1b',
                           input=X,
                           filter=W_1b,
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    max_1b = tf.nn.max_pool(name='max_1b',
                        value=conv_1b,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    activ_1b = tf.nn.elu(name='active_1b',
                         features=max_1b)
    norm_1b = tf.layers.batch_normalization(
                    name='norm_1b',
                    inputs=activ_1b,
                    axis=3,
                    training=(mode==tf.estimator.ModeKeys.TRAIN))

    # conv layer 2
    W_2a = tf.get_variable(name='W_2a',
                           shape=[5, 5, conv_k_chs[0], conv_k_chs[1]],
                           initializer=tf.contrib.layers.xavier_initializer())
    conv_2a = tf.nn.conv2d(name='conv_2a',
                           input=norm_1a,
                           filter=W_2a,
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    max_2a = tf.nn.max_pool(name='max_2a',
                        value=conv_2a,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    activ_2a = tf.nn.elu(name='activ_2a',
                         features=max_2a)
    norm_2a = tf.layers.batch_normalization(
                        name='norm_2a',
                        inputs=activ_2a,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    W_2b = tf.get_variable(name='W_2b',
                           shape=[5, 5, conv_k_chs[0], conv_k_chs[1]],
                           initializer=tf.contrib.layers.xavier_initializer())
    conv_2b = tf.nn.conv2d(name='conv_2b',
                           input=norm_1b,
                           filter=W_2b,
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    max_2b = tf.nn.max_pool(name='max_2b',
                            value=conv_2b,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
    activ_2b = tf.nn.elu(name='activ_2b',
                         features=max_2b)
    norm_2b = tf.layers.batch_normalization(
                        name='norm_2b',
                        inputs=activ_2b,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    # conv layer 3
    input_3 = tf.concat(name='input_3', 
                        values=[norm_2a, norm_2b], 
                        axis=3)
    W_3 = tf.get_variable(name='W_3',
                          shape=[5, 5, conv_k_chs[1]*2, conv_k_chs[2]],
                          initializer=tf.contrib.layers.xavier_initializer())
    conv_3 = tf.nn.conv2d(name='conv_3',
                          input=input_3,
                          filter=W_3,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    activ_3 = tf.nn.elu(name='activ_3',
                        features=conv_3)
    norm_3 = tf.layers.batch_normalization(
                        name='norm_3',
                        inputs=activ_3,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    # conv layer 4
    W_4a = tf.get_variable(name='W_4a',
                           shape=[5, 5, conv_k_chs[2], conv_k_chs[3]],
                           initializer=tf.contrib.layers.xavier_initializer())
    conv_4a = tf.nn.conv2d(name='conv_4a',
                           input=activ_3,
                           filter=W_4a,
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    activ_4a = tf.nn.elu(name='activ_4a',
                         features=conv_4a)
    norm_4a = tf.layers.batch_normalization(
                        name='norm_4a',
                        inputs=activ_4a,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    W_4b = tf.get_variable(name='W_4b',
                           shape=[5, 5, conv_k_chs[2], conv_k_chs[3]],
                           initializer=tf.contrib.layers.xavier_initializer())
    conv_4b = tf.nn.conv2d(name='conv_4b',
                           input=activ_3,
                           filter=W_4b,
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    activ_4b = tf.nn.elu(name='activ_4b',
                         features=conv_4b)
    norm_4b = tf.layers.batch_normalization(
                        name='norm_4b',
                        inputs=activ_4b,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    # conv layer 5
    W_5a = tf.get_variable(name='W_5a',
                           shape=[5, 5, conv_k_chs[3], conv_k_chs[4]],
                           initializer=tf.contrib.layers.xavier_initializer())
    conv_5a = tf.nn.conv2d(name='conv_5a',
                           input=norm_4a,
                           filter=W_5a,
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    activ_5a = tf.nn.elu(name='activ_5a',
                         features=conv_5a)
    norm_5a = tf.layers.batch_normalization(
                        name='norm_5a',
                        inputs=activ_5a,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    W_5b = tf.get_variable(name='W_5b',
                           shape=[5, 5, conv_k_chs[3], conv_k_chs[4]],
                           initializer=tf.contrib.layers.xavier_initializer())
    conv_5b = tf.nn.conv2d(name='conv_5b',
                           input=norm_4b,
                           filter=W_5b,
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    activ_5b = tf.nn.elu(name='activ_5b',
                         features=conv_5b)
    norm_5b = tf.layers.batch_normalization(
                        name='norm_5b',
                        inputs=activ_5b,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    # fully-connected layer 1
    input_fc_1 = tf.concat([norm_5a, norm_5b], axis=3)
    fc_1_dim = (75**2) * conv_k_chs[4] * 2
    input_fc_1 = tf.reshape(name='input_fc_1',
                            tensor=input_fc_1, 
                            shape=[-1, fc_1_dim])
    W_fc_1 = tf.get_variable(name='W_fc_1',
                             shape=[fc_1_dim, fc1_units],
                             initializer=tf.contrib.layers.xavier_initializer())
    b_fc_1 = tf.Variable(name='b_fc_1',
                         initial_value=tf.random_uniform(shape=[fc1_units, ],
                                                         maxval=0.1))
    logits_fc_1 = tf.matmul(input_fc_1, W_fc_1) + b_fc_1
    dropout_fc_1 = tf.layers.dropout(name='dropout_fc_1',
                                     inputs=logits_fc_1,
                                     rate=dropout_rate,
                                     training=is_training)
    activ_fc_1 = tf.nn.elu(name='activ_fc_1',
                        features=dropout_fc_1)
    norm_fc_1 = tf.layers.batch_normalization(
                        name='norm_fc_1',
                        inputs=activ_fc_1,
                        axis=1,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    # fully-connected layer 2
    W_fc_2 = tf.get_variable(name='W_fc_2',
                             shape=[fc1_units, fc2_units],
                             initializer=tf.contrib.layers.xavier_initializer())
    b_fc_2 = tf.Variable(name='b_fc_2',
                         initial_value=tf.random_uniform(shape=[fc2_units, ],
                                                         maxval=0.1))
    logits_fc_2 = tf.matmul(norm_fc_1, W_fc_2) + b_fc_2
    dropout_fc_2 = tf.layers.dropout(name='dropout_fc_2',
                                 inputs=logits_fc_2,
                                 rate=dropout_rate,
                                 training=is_training)
    activ_fc_2 = tf.nn.elu(dropout_fc_2)
    norm_fc_2 = tf.layers.batch_normalization(
                        name='norm_fc_2',
                        inputs=activ_fc_2,
                        axis=1,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    # softmax layer
    W_sm = tf.get_variable(name='W_sm',
                           shape=[fc2_units, 2],
                           initializer=tf.contrib.layers.xavier_initializer())
    b_sm = tf.Variable(name='b_sm',
                       initial_value=tf.random_uniform(shape=[2, ],
                                                       maxval=0.1))
    logits_sm = tf.matmul(norm_fc_2, W_sm) + b_sm
    softmax = tf.nn.softmax(logits_sm)

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


def main():
    hyperparameters = {'learning_rate': 1e-2,
                       'moment1': 0.9,
                       'moment2': 0.999,
                       'in_channels': 3,
                       'conv_k_chs': [32, 32, 16, 16, 8],
                       'fc1_units': 256,
                       'fc2_units': 32}