import tensorflow as tf
import numpy as np

def weight_variable_refine(shape, scale, name=None):
	init = tf.random_normal(shape=shape) / scale
	return tf.Variable(init, name=name)

def bias_variable(b_init, shape, name=None):
	init = tf.constant(b_init, shape=shape)
	return tf.Variable(init, name=name)

def conv2d(x, w, stride, pad='SAME'):
	return tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding=pad)

def max_pool(x, kernel, stride, pad='VALID'):
	return tf.nn.max_pool(x, ksize=[1,kernel,kernel,1], strides=[1,stride,stride,1], padding=pad)

def offset_variable(shape, name=None):
	init = tf.constant(0.0, shape=shape)
	return tf.Variable(init, name=name)

def scale_variable(shape, name=None):
	init = tf.constant(1.0, shape=shape)
	return tf.Variable(init, name=name)

class detect_48Net:
	def __init__(self, inputs, targets, keep_prob, lr=1e-3, bias_init=0.0, reg=5e-3, Train=True, BN_list1=[0,0], BN_list2=[0,0]):
		with tf.variable_scope('48det_'):
			with tf.device('/gpu:0'):
				with tf.variable_scope('conv1'):
					# 5x5 filters / stride 1 / 64 feature maps
					self.w_conv1 = weight_variable_refine([5,5,3,64], np.sqrt(5*5*3/2.0), 'w1')
					self.b_conv1 = bias_variable(bias_init, [64], 'b1')
					self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)
					
					# max_pool with 3 kernel size and stide 2
					self.h_pool1 = max_pool(self.h_conv1, 3, 2)
					self.axis1 = list(range(len(self.h_pool1.get_shape()) - 1))
					
					self.of_conv1 = offset_variable([64], 'offset1')
					self.sc_conv1 = scale_variable([64], 'scale1')
					
					# batch_norm on the max_pool outputs
					if Train:
						self.mean_wconv1, self.var_wconv1 = tf.nn.moments(self.h_pool1, axes=self.axis1)
						self.mean_wconv1 = tf.identity(self.mean_wconv1, 'mean_wconv1')
						self.var_wconv1 = tf.identity(self.var_wconv1, 'var_wconv1')
					else:
						self.mean_wconv1, self.var_wconv1 = BN_list1[0].astype(np.float32), BN_list1[1].astype(np.float32)
						self.mean_wconv1 = tf.identity(self.mean_wconv1, 'mean_wconv1')
						self.var_wconv1 = tf.identity(self.var_wconv1, 'var_wconv1')

					self.h_pool1_bn = tf.nn.batch_normalization(self.h_pool1, self.mean_wconv1, self.var_wconv1,
															   self.of_conv1, self.sc_conv1, variance_epsilon=1e-3)

				with tf.variable_scope('conv2'):
					# 5x5 filters / stride 1 / 64 feature maps
					self.w_conv2 = weight_variable_refine([5,5,64,64], np.sqrt(5*5*64/2.0), 'w2')
					self.b_conv2 = bias_variable(bias_init, [64], 'b2')
					
					# self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.w_conv2, 1) + self.b_conv2)
					self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1_bn, self.w_conv2, 1) + self.b_conv2)
					self.axis2 = list(range(len(self.h_conv2.get_shape()) - 1))
					
					self.of_conv2 = offset_variable([64], 'offset2')
					self.sc_conv2= scale_variable([64], 'scale2')
					
					# batch_norm on the h_conv2 layer
					if Train:
						self.mean_wconv2, self.var_wconv2 = tf.nn.moments(self.h_conv2, axes=self.axis2)
						self.mean_wconv2 = tf.identity(self.mean_wconv2, 'mean_wconv2')
						self.var_wconv2 = tf.identity(self.var_wconv2, 'var_wconv2')
						
					else:
						self.mean_wconv2, self.var_wconv2 = BN_list2[0].astype(np.float32), BN_list2[1].astype(np.float32)
						self.mean_wconv2 = tf.identity(self.mean_wconv2, 'mean_wconv2')
						self.var_wconv2 = tf.identity(self.var_wconv2, 'var_wconv2')

					self.h_conv2_bn = tf.nn.batch_normalization(self.h_conv2, self.mean_wconv2, self.var_wconv2,
																	self.of_conv2, self.sc_conv2, variance_epsilon=1e-3)
					# max_pool with 3 kernel size and stride 2
					# self.h_pool2 = max_pool(self.h_conv2, 3, 2)
					self.h_pool2 = max_pool(self.h_conv2_bn, 3, 2)

				with tf.variable_scope('fc_conv1'):
					# 12x12 filters / stide 1 for elastic evaluation
					self.w_fc_conv1 = weight_variable_refine([11,11,64,256], np.sqrt(12*12*64/2.0), 'w3')
					self.b_fc_conv1 = bias_variable(bias_init, [256], 'b3')
					self.h_fc_conv1 = tf.nn.relu(conv2d(self.h_pool2, self.w_fc_conv1, 1, pad='VALID') + self.b_fc_conv1)
					self.h_fc_conv1_drop = tf.nn.dropout(self.h_fc_conv1, keep_prob)

				with tf.variable_scope('fc_conv2'):
					self.w_fc_conv2 = weight_variable_refine([1,1,256,1], np.sqrt(256.0), 'w4')
					self.b_fc_conv2 = bias_variable(bias_init, [1], 'b4')
					self.h_fc_conv2 = tf.nn.sigmoid(conv2d(self.h_fc_conv1_drop, self.w_fc_conv2, 1) + self.b_fc_conv2)

				with tf.variable_scope('prediction'):
					self.prediction = self.h_fc_conv2
					self.fc2_shape = tf.concat([[-1], [tf.reduce_prod(tf.slice(tf.shape(self.h_fc_conv2),[1],[3]), 0)]], 0)
					self.h_fc_conv2_flat = tf.reshape(self.prediction, self.fc2_shape)
				with tf.variable_scope('loss'):
					self.w_loss = tf.nn.l2_loss(self.w_conv1) + tf.nn.l2_loss(self.w_conv2) + tf.nn.l2_loss(self.w_fc_conv1) + tf.nn.l2_loss(self.w_fc_conv2)

					self.loss = tf.reduce_mean(tf.add(-tf.reduce_sum(targets * tf.log(self.h_fc_conv2_flat + 1e-9), 1), 
								-tf.reduce_sum((1 - targets) * tf.log(1 - self.h_fc_conv2_flat + 1e-9), 1)), name='loss')
					self.regularization_loss = tf.add(self.loss, reg * self.w_loss, name='regularization_loss')
				with tf.variable_scope('training'):	
					self.train_step = tf.train.AdamOptimizer(lr).minimize(self.regularization_loss)

class calib_48Net:
	def __init__(self, inputs, targets, keep_prob, lr=5e-3, bias_init=0.0, reg=1e-3, Train=True, BN_list1=[0,0], BN_list2=[0,0]):
		with tf.variable_scope('48cal_'):
			with tf.variable_scope('conv1'):
				# 5x5 filers, stride 1, 64 feature maps
				self.w_conv1 = weight_variable_refine([5,5,3,64], np.sqrt(5*5*3/2.0), 'w1')
				self.b_conv1 = bias_variable(bias_init, [64], 'b1')
				self.h_conv1 = tf.nn.relu(conv2d(inputs, self.w_conv1, 1) + self.b_conv1)
				# 3x3 kernel, stride 2
				self.h_pool1 = max_pool(self.h_conv1, 3, 2)

				self.of_conv1 = offset_variable([64], 'offset1')
				self.sc_conv1 = scale_variable([64], 'scale1')

				self.axis1 = list(range(len(self.h_pool1.get_shape()) - 1))
				if Train:
					self.mean_conv1, self.var_conv1 = tf.nn.moments(self.h_pool1, axes=self.axis1)
					self.mean_conv1 = tf.identity(self.mean_conv1, 'mean_conv1')
					self.var_conv1 = tf.identity(self.var_conv1, 'var_conv1')
				else:
					self.mean_conv1, self.var_conv1 = BN_list1[0].astype(np.float32), BN_list1[1].astype(np.float32)
				self.h_pool1_bn = tf.nn.batch_normalization(self.h_pool1, self.mean_conv1, self.var_conv1,
															self.of_conv1, self.sc_conv1, variance_epsilon=1e-7)
			with tf.variable_scope('conv2'):
				# 5x5 filters, stide 1, 64 feature maps
				self.w_conv2 = weight_variable_refine([5,5,64,64], np.sqrt(5*5*64/2.0), 'w2')
				self.b_conv2 = bias_variable(bias_init, [64], 'b2')
				self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1_bn, self.w_conv2, 1) + self.b_conv2)

				self.of_conv2 = offset_variable([64], 'offset2')
				self.sc_conv2 = scale_variable([64], 'scale2')

				self.axis2 = list(range(len(self.h_conv2.get_shape()) - 1))

				if Train:
					self.mean_conv2, self.var_conv2 = tf.nn.moments(self.h_conv2, axes=self.axis2)
					self.mean_conv2 = tf.identity(self.mean_conv2, 'mean_conv2')
					self.var_conv2 = tf.identity(self.var_conv2, 'var_conv2')
				else:
					self.mean_conv2, self.var_conv2 = BN_list2[0].astype(np.float32), BN_list2[1].astype(np.float32)
				self.h_conv2_bn = tf.nn.batch_normalization(self.h_conv2, self.mean_conv2, self.var_conv2,
															self.of_conv2, self.sc_conv2, variance_epsilon=1e-7)
				self.h_pool2 = max_pool(self.h_conv2_bn, 3, 2)
				self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 11*11*64])
			with tf.variable_scope('fc1'):
				# 256 nodes
				self.w_fc1 = weight_variable_refine([11*11*64,256], np.sqrt(11*11*64/2.0), 'w3')
				self.b_fc1 = bias_variable(bias_init, [256], 'b3')
				self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.w_fc1) + self.b_fc1)
				self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob)

			with tf.variable_scope('fc2'):
				# 45 classes
				self.w_fc2 = weight_variable_refine([256,45], np.sqrt(256/2.0), 'w4')
				self.b_fc2 = bias_variable(bias_init, [45], 'b4')
				self.h_fc2 = tf.matmul(self.h_fc1_drop, self.w_fc2) + self.b_fc2

			with tf.variable_scope('prediction'):
				self.prediction = tf.nn.softmax(self.h_fc2, name='pred')
			with tf.variable_scope('loss'):
				self.w_loss = tf.nn.l2_loss(self.w_conv1) + tf.nn.l2_loss(self.w_conv2) + \
								tf.nn.l2_loss(self.w_fc1) + tf.nn.l2_loss(self.w_fc2)
				self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.h_fc2,
																	labels=targets,
																	name='loss'))
				self.regularization_loss = tf.add(self.loss, reg * self.w_loss, name='regularization_loss')
			with tf.variable_scope('training'):
				self.train_step = tf.train.AdamOptimizer(lr).minimize(self.regularization_loss)