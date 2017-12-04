import tensorflow as tf
import numpy as np


##### reference:
#    http://ruishu.io/2016/12/27/batchnorm/

class BasicCNN:
    def __init__(self, 
                 name='BasicCNN',
                 input_shape=[None, 75, 75, 2],
                 output_shape=[None, 2],
                 conv_layers=2, 
                 conv_size=[5, 5],
                 conv_chs=[32, 16],
                 fc_layers=1,
                 fc_size=[2]):
        '''
        define comp graph w/ placeholders

        separate train and validation
        '''
        pass
        '''
        ----Arguments----
        - input_shape: Shape of input data
          expects [None, H, W, C]
        - output_shape: Shape of output 
        - conv_layers: Number of modules of
          convolution -> maxpool -> activation
        - conv_size: Filter size for convolutions
        - conv_chs: Output channel sizes for 
          each convolution module
        - fc_layers: Number of fully-connected
          layers after convolutions
        - fc_size: Number of hidden units in
          each fully-connected layer
        '''

        self.name = name

        with tf.variable_scope('input'):
            X = tf.placeholder(name='X',
                               dtype=tf.float32, 
                               shape=input_shape)
            is_training = tf.placeholder(name='is_training',
                                         dtype=tf.bool,
                                         shape=[1])

        with tf.variable_scope('convolutions'):
            conv_output = X

            for layer_id, ch_size in zip(xrange(conv_layers), conv_chs):
                conv_output = self._conv_module(input_layer=conv_output,
                                                conv_size=conv_size,
                                                out_chs=ch_size,
                                                iteration=layer_id)
                conv_output = self._batch_normalize(input_layer=conv_output,
                                                    is_training=is_training)

        with tf.variable_scope('dense'):
            dense_dim = tf.shape(conv_output)[1] * \
                        tf.shape(conv_output)[2] * \
                        tf.shape(conv_output)[3]
            dense_output = tf.reshape(conv_output, shape=[-1, dense_dim])

            for layer_id, layer_size in zip(xrange(fc_layers), fc_size):
                is_final_layer = tf.cast(layer_id == fc_layers - 1, tf.bool)
                dense_output = self._fc_layer(input_layer=dense_output,
                                              out_size=layer_size,
                                              iteration=layer_id,
                                              final=is_final_layer)

        with tf.variable_scope('predictions'):
            pred_probs = tf.nn.softmax(name='softmax',
                                       logits=dense_output)
            pred_labels = tf.argmax(name='pred_labels',
                                    input=pred_probs,
                                    axis=1)

            self._pred_probs = pred_probs
            self._pred_labels = pred_labels

        with tf.variable_scope('loss'):
            y = tf.placeholder(name='y',
                               dtype=tf.int32,
                               shape=[None])
            y_onehot = tf.one_hot(indices=y, depth=2, axis=1)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot,
                                                   logits=dense_output)
            self._loss = loss

        with tf.variable_scope('optimize'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            min_op = optimizer.minimize(loss=loss,
                               global_step=tf.train.get_global_step())
            self._min_op = min_op


    def _batch_normalize(input_layer, is_training):
        '''
        batch normalization with trained post-normalization
        mean/var params 
        '''
        output = tf.contrib.layers.batch_norm(inputs=input_layer,
                                              center=True,
                                              scale=True,
                                              is_training=is_training)
        return(output)

    
    def _conv_module(input_layer, conv_size, out_chs, iteration):
        '''
        factory function for convolutional layers
        '''
        in_chs = tf.shape(input_layer)[3]
        conv_w = tf.get_variable(name='conv_w' + str(iteration),
                      shape=[conv_size, 
                             conv_size, 
                             in_chs, 
                             out_chs],
                      initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(name='conv' + str(iteration),
                            input=input_layer,
                            filter=conv_w,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        maxpool = tf.nn.max_pool(name='max' + str(iteration),
                                 value=conv,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME')
        activ = tf.nn.elu(maxpool)

        return(activ)


    def _fc_layer(input_layer, out_size, iteration, 
                  final=tf.cast(False, dtype=tf.bool)):
        '''
        write in dropout
        '''
        pass
        '''
        factory function for fully-connected layers
        '''
        in_size = tf.shape(input_layer)[1]
        fc_w = tf.get_variable(name='fc_w' + str(iteration),
                    shape=[in_size, out_size],
                    initializer=tf.contrib.layers.xavier_initializer())
        fc_b = tf.Variable(name='fc_b' + str(iteration),
                    initial_value=tf.random_uniform(shape=[out_size, ],
                                                    maxval=0.1))
        fc_logits = tf.matmul(input_layer, fc_w) + b

        fc_activ = tf.cond(pred=final, 
                           true_fn=lambda: tf.identity(
                                name='fc_logits',
                                input=fc_logits),
                           false_fn=lambda: tf.tf.nn.elu(
                                name='fc_activ' + str(iteration),
                                features=fc_logits))

        return(fc_activ)


    def _loss(self, X, y):
        '''
        return loss computation
        '''
        pass


    def _optimize(self, X, y):
        '''
        return minimize operation
        '''
        pass


    def _summarize(self, X, y):
        '''
        save training curve
        '''
        pass


    def predict(self, X):
        '''
        return feed-forward computation
        '''
        pass


    def train(self, X_train, y_train, 
              X_test, y_test, save_steps=100, 
              model_dir='./' + self.name):
        '''
        begin tf.Session()
        call optimize, loss, summarize
        save loss and metrics each save_steps iterations
        '''
        pass
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = self._min_op

