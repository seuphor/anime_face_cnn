import tensorflow as tf
import numpy as np

w_std = 0.1
b_init = 0.0
lr = 5e-2

def weight_variable(shape, scale, name=None):
    init = tf.truncated_normal(shape=shape, stddev=w_std) / scale
    return tf.Variable(init, name=name)

def bias_variable(shape, name=None):
    init = tf.constant(value=b_init, shape=shape)
    return tf.Variable(init, name=name)

def conv2d(x, W, stride, pad = 'SAME'):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding=pad)

def max_pool(x, kersize, stride, pad = 'SAME'):
    return tf.nn.max_pool(x, ksize=[1,kersize,kersize,1], strides=[1,stride,stride,1], padding=pad)

class detect_12Net:
    def __init__(self, inputs, targets):
        
        with tf.variable_scope('12det_'):
            with tf.device('/gpu:0'):
                # conv layer 1
                self.w_conv1 = weight_variable([3,3,3,16], tf.sqrt(3*3*3.0 / 2.0), 'w1')
                self.b_conv1 = bias_variable([16], 'b1')
                self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)
                
                # pooling layer 1
                self.h_pool1 = max_pool(self.h_conv1, 3, 2)
                
                # fully conv layer 1
                self.w_conv2 = weight_variable([6, 6, 16, 16], tf.sqrt(6*6*16.0 / 2.0), 'w2')
                self.b_conv2 = bias_variable([16], 'b2')
                
                # 2 = window_size / 2 cuz we do 1 max pool
                self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.w_conv2, 2, pad='VALID') + self.b_conv2)   
                
                #fully conv layer 2
                self.w_conv3 = weight_variable([1,1,16,1], tf.sqrt(1*1*16.0 / 2.0), "w3")
                self.b_conv3 = bias_variable([1],"b3")
                self.h_conv3 = tf.nn.sigmoid(conv2d(self.h_conv2, self.w_conv3, 1) + self.b_conv3)
                
                self.conv3_shape = tf.concat([[-1],[tf.reduce_prod(tf.slice(tf.shape(self.h_conv3),[1],[3]),0)]], 0)
                self.h_conv3_reshaped = tf.reshape(self.h_conv3,self.conv3_shape)
            
        self.prediction = self.h_conv3
        self.prediction_flatten = self.h_conv3_reshaped
        with tf.device('/gpu:0'):
            self.loss = tf.reduce_mean(tf.add(-tf.reduce_sum(targets * tf.log(self.prediction_flatten + 1e-9), 1), -tf.reduce_sum((1-targets) * tf.log(1-self.prediction_flatten + 1e-9), 1)))
            self.train_step = tf.train.AdadeltaOptimizer(lr).minimize(self.loss)
    