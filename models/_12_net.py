import tensorflow as tf
import numpy as np

w_std = 0.1
#b_init = 0.0
#lr = 5e-2

def weight_variable(shape, scale, name=None):
    init = tf.truncated_normal(shape=shape, stddev=w_std) / scale
    return tf.Variable(init, name=name)

def weight_variable_refine(shape, scale, name=None):

    init = tf.random_normal(shape=shape) / scale
    return tf.Variable(init, name=name)

def bias_variable(b_init, shape, name=None):
    init = tf.constant(value=b_init, shape=shape)
    return tf.Variable(init, name=name)

def conv2d(x, W, stride, pad = 'SAME'):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding=pad)

def max_pool(x, kersize, stride, pad = 'SAME'):
    return tf.nn.max_pool(x, ksize=[1,kersize,kersize,1], strides=[1,stride,stride,1], padding=pad)

class detect_12Net:
    def __init__(self, inputs, targets, keep_prob, learning_rate, bias_init=0.0):
        
        with tf.variable_scope('12det_'):

                with tf.device('/gpu:0'):

                    # conv layer 1
                    with tf.variable_scope('conv1'):
                        self.w_conv1 = weight_variable_refine([3,3,3,16], np.sqrt(3*3*3/2.0), 'w1')
                        self.b_conv1 = bias_variable(bias_init, [16], 'b1')
                        self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)
                        
                        # pooling layer 1
                        self.h_pool1 = max_pool(self.h_conv1, 3, 2)

                        # tf.summary
                        tf.summary.histogram('w1', self.w_conv1)
                        tf.summary.histogram('b1', self.b_conv1)
                    
                    with tf.variable_scope('fc_conv1'):
                        # fully conv layer 1
                        self.w_conv2 = weight_variable_refine([6, 6, 16, 16], np.sqrt(6*6*16/2.0), 'w2')
                        self.b_conv2 = bias_variable(bias_init, [16], 'b2')
                        
                        # 2 = window_size / 2 cuz we do 1 max pool
                        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.w_conv2, 2, pad='VALID') + self.b_conv2)
                        self.h_conv2_drop = tf.nn.dropout(self.h_conv2, keep_prob)

                        # tf.summary
                        tf.summary.histogram('w2', self.w_conv2)
                        tf.summary.histogram('b2', self.b_conv2)   
                    
                    with tf.variable_scope('fc_conv2'):
                        #fully conv layer 2
                        self.w_conv3 = weight_variable_refine([1,1,16,1], np.sqrt(16.0), "w3")
                        self.b_conv3 = bias_variable(bias_init, [1],"b3")
                        self.h_conv3 = tf.nn.sigmoid(conv2d(self.h_conv2_drop, self.w_conv3, 1) + self.b_conv3)

                        # tf.summary
                        tf.summary.histogram('w3', self.w_conv3)
                        tf.summary.histogram('b3', self.b_conv3)
                    
                        self.conv3_shape = tf.concat([[-1],[tf.reduce_prod(tf.slice(tf.shape(self.h_conv3),[1],[3]),0)]], 0)
                        self.h_conv3_reshaped = tf.reshape(self.h_conv3,self.conv3_shape)
                
                    with tf.variable_scope('prediction'):    
                        self.prediction = self.h_conv3
                        self.prediction_flatten = self.h_conv3_reshaped
                        self.pred_summary = tf.identity(self.prediction_flatten, "pred_summary")

                        # tf.summary
                        tf.summary.histogram('pred_summary', self.pred_summary)

                    with tf.variable_scope('loss'):
                        self.w_loss = tf.nn.l2_loss(self.w_conv1) + tf.nn.l2_loss(self.w_conv2) + tf.nn.l2_loss(self.w_conv3)
                        self.pred_loss = tf.add(-tf.reduce_sum(targets * tf.log(self.prediction_flatten + 1e-9), 1), 
                                                          -tf.reduce_sum((1-targets) * tf.log(1-self.prediction_flatten + 1e-9), 1))

                        self.loss = tf.reduce_mean(self.pred_loss, name="loss")
                        self.regularization_loss = tf.reduce_mean(tf.add(self.pred_loss, self.w_loss * 1e-3), name="regularization_loss")
                        # tf.summary
                        tf.summary.scalar('loss', self.loss)

                    with tf.variable_scope('training'):
                        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.regularization_loss)  #AdadeltaOptimizer
                        
                self.merged = tf.summary.merge_all()

class calib_12Net:
    def __init__(self, inputs, targets, keep_prob, lr=5e-2, bias_init=0.0, reg=1e-3):
        with tf.variable_scope('12cal_'):

            with tf.device('/gpu:0'):
                with tf.variable_scope('conv1'):
                    # conv1
                    self.w_conv1 = weight_variable_refine([3,3,3,16], tf.sqrt(3*3*3.0/2), 'w1')
                    self.b_conv1 = bias_variable(bias_init, [16], 'b1')

                    self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)
                    self.h_pool1 = max_pool(self.h_conv1, 3, 2)

                    tf.summary.histogram('w1', self.w_conv1)
                    tf.summary.histogram('b1', self.b_conv1)

                with tf.variable_scope('fc1'):
                    # fc1
                    self.w_fc1 = weight_variable_refine([6*6*16,128], tf.sqrt(6*6*16.0/2), 'w2')
                    self.b_fc1 = bias_variable(bias_init, [128], 'b2')
                    
                    self.h_pool1_flat = tf.reshape(self.h_pool1, [-1, 6*6*16])
                    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool1_flat, self.w_fc1) + self.b_fc1)
                    self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob)
                    tf.summary.histogram('w2', self.w_fc1)
                    tf.summary.histogram('b2', self.b_fc1)

                with tf.variable_scope('fc2'):
                    # fc_conv2
                    self.w_fc2 = weight_variable_refine([128,45], tf.sqrt(128.0), 'w3')
                    self.b_fc2 = bias_variable(bias_init, [45], 'b3')
                    self.h_fc2 = tf.matmul(self.h_fc1_drop, self.w_fc2) + self.b_fc2

                    tf.summary.histogram('w3', self.w_fc2)
                    tf.summary.histogram('b3', self.b_fc2)

                with tf.variable_scope('prediction'):
                    # softmax prediction
                    self.prediction = tf.nn.softmax(self.h_fc2, name='pred')

                    tf.summary.histogram('pred', self.prediction)

                with tf.variable_scope('loss'):
                    # loss
                    self.w_loss = tf.nn.l2_loss(self.w_conv1) + tf.nn.l2_loss(self.w_fc1) + tf.nn.l2_loss(self.w_fc2)

                    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.h_fc2, labels=targets), name="loss")
                    self.regularization_loss = tf.add(self.loss, reg * self.w_loss, name="regularization_loss")

                    tf.summary.scalar('loss', self.loss)

                with tf.variable_scope('training'):
                    # optimizer
                    self.train_step = tf.train.AdamOptimizer(lr).minimize(self.regularization_loss)

                self.merged = tf.summary.merge_all()

