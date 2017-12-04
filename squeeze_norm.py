import tensorflow as tf

'''
Training notes:
- Batch size = 32, learning rate = 1e-4 
  produced > linear.py results
- Increased conv_chs, sqz, exp1, exp3 params
  improved performance
'''


def fire_module(inputs, input_ch, sqz_param, exp1_param, exp3_param, iteration, mode):
    '''
    fire module + batch normalization to 
    speed up training

    assumes input image size of Nx75x75xC
    for N images
    for C channnels
    '''
    # squeeze
    W_sqz = tf.get_variable(name='W_sqz_'+str(iteration),
                            shape=[1, 1, input_ch, sqz_param],
                            initializer=tf.contrib.layers.xavier_initializer())
    conv_sqz = tf.nn.conv2d(name='conv_sqz_'+str(iteration),
                            input=inputs,
                            filter=W_sqz,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
    activ_sqz = tf.nn.elu(name='activ_sqz_'+str(iteration),
                          features=conv_sqz)
    norm_sqz = tf.layers.batch_normalization(
                          name='norm_sqz_'+str(iteration),
                          inputs=activ_sqz,
                          axis=3,
                          training=(mode==tf.estimator.ModeKeys.TRAIN))

    # expand 1x1
    W_exp1 = tf.get_variable(name='W_exp1_'+str(iteration),
                             shape=[1, 1, sqz_param, exp1_param],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_exp1 = tf.nn.conv2d(name='conv_exp1_'+str(iteration),
                             input=norm_sqz,
                             filter=W_exp1,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
    activ_exp1 = tf.nn.elu(name='activ_exp1',
                           features=conv_exp1)
    norm_exp1 = tf.layers.batch_normalization(
                          name='norm_exp1_'+str(iteration),
                          inputs=activ_exp1,
                          axis=3,
                          training=(mode==tf.estimator.ModeKeys.TRAIN))

    # expand 3x3
    W_exp3 = tf.get_variable(name='W_exp3_'+str(iteration),
                             shape=[3, 3, sqz_param, exp3_param],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_exp3 = tf.nn.conv2d(name='conv_exp3_'+str(iteration),
                             input=norm_sqz,
                             filter=W_exp3,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
    activ_exp3 = tf.nn.elu(name='activ_exp3_'+str(iteration),
                           features=conv_exp3)
    norm_exp3 = tf.layers.batch_normalization(
                          name='norm_exp3_'+str(iteration),
                          inputs=activ_exp3,
                          axis=3,
                          training=(mode==tf.estimator.ModeKeys.TRAIN))

    fire_module = tf.concat(name='fire_'+str(iteration),
                            values=[norm_exp1, norm_exp3], 
                            axis=3)

    return(fire_module)


def squeezenet_function(features, labels, mode, params):
    '''
    assumes features are N x 75 x 75 x C
    for batch size N 
    for C channels
    '''
    learning_rate = params['learning_rate']
    moment1 = params['moment1']
    moment2 = params['moment2']
    conv_chs = params['conv_chs']

    # fire module params
    n_fire_mod = params['n_fire_mod']
    sqz_param = params['sqz_param']
    exp1_param = params['exp1_param']
    exp3_param = params['exp3_param']

    X = tf.cast(features['x'], dtype=tf.float32)

    W_1 = tf.get_variable(name='W_1',
                          shape=[5, 5, 2, conv_chs[0]],
                          initializer=tf.contrib.layers.xavier_initializer())
    conv_1 = tf.nn.conv2d(name='conv_1',
                          input=X,
                          filter=W_1,
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    max_1 = tf.nn.max_pool(name='max_1',
                           value=conv_1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    activ_1 = tf.nn.elu(name='activ_1',
                        features=max_1)
    norm_1 = tf.layers.batch_normalization(
                        name='norm_1',
                        inputs=activ_1,
                        axis=3,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    fire_output_1 = norm_1
    input_ch = conv_chs[0]
    for fire_id in xrange(n_fire_mod[0]):
        fire_output_1 = fire_module(inputs=fire_output_1,
                                    input_ch=input_ch,
                                    sqz_param=sqz_param[0],
                                    exp1_param=exp1_param[0],
                                    exp3_param=exp3_param[0],
                                    iteration=fire_id,
                                    mode=mode)
        input_ch = exp1_param[0] + exp3_param[0]
    max_2 = tf.nn.max_pool(name='max_2',
                           value=fire_output_1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    fire_output_2 = max_2
    for fire_id in xrange(n_fire_mod[0], 
                          n_fire_mod[0]+n_fire_mod[1]):
        fire_output_2 = fire_module(inputs=fire_output_2,
                                    input_ch=input_ch,
                                    sqz_param=sqz_param[1],
                                    exp1_param=exp1_param[1],
                                    exp3_param=exp3_param[1],
                                    iteration=fire_id,
                                    mode=mode)
        input_ch = exp1_param[1] + exp3_param[1]
    max_3 = tf.nn.max_pool(name='max_3',
                           value=fire_output_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    fire_output_3 = max_3
    for fire_id in xrange(n_fire_mod[0]+n_fire_mod[1], sum(n_fire_mod)):
        fire_output_3 = fire_module(inputs=fire_output_3,
                                    input_ch=input_ch,
                                    sqz_param=sqz_param[2],
                                    exp1_param=exp1_param[2],
                                    exp3_param=exp3_param[2],
                                    iteration=fire_id,
                                    mode=mode)
        input_ch = exp1_param[2] + exp3_param[2]

    W_2 = tf.get_variable(name='W_2',
                          shape=[5, 5, exp1_param[2]+exp3_param[2], conv_chs[1]],
                          initializer=tf.contrib.layers.xavier_initializer())
    conv_2 = tf.nn.conv2d(name='conv_2',
                          input=fire_output_3,
                          filter=W_2,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    activ_2 = tf.nn.elu(name='activ_2',
                        features=conv_2)
    avg_pool = tf.reduce_mean(name='avg_pool',
                              input_tensor=conv_2,
                              axis=[1, 2])
    norm_2 = tf.layers.batch_normalization(
                        name='norm_2',
                        inputs=avg_pool,
                        axis=1,
                        training=(mode==tf.estimator.ModeKeys.TRAIN))

    W_last = tf.get_variable(name='W_last',
                             shape=[conv_chs[1], 2],
                             initializer=tf.contrib.layers.xavier_initializer())
    b_last = tf.Variable(name='b_last',
                         initial_value=tf.random_uniform(shape=[2, ],
                                                         maxval=0.1))
    logits = tf.matmul(norm_2, W_last) + b_last
    predictions = tf.nn.softmax(name='softmax', logits=logits)


    if mode == tf.estimator.ModeKeys.TRAIN:
        one_hot_labels = tf.one_hot(indices=labels, depth=2)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels,
                                               logits=logits)
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=moment1,
                                      beta2=moment2)
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

    

def main():

    hyperparameters = {'learning_rate': 1e-3,
                       'moment1': 0.9,
                       'moment2': 0.999,
                       'conv_chs': [96, 10],
                       'n_fire_mod': [3, 4, 1],
                       'sqz_param': [4, 8, 16],
                       'exp1_param': [8, 16, 32],
                       'exp3_param': [8, 16, 32]}