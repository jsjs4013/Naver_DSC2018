import math
import numpy as np 
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

    
# Anomaly detection을 하기 위해 auencoder를 0에 대하여 학습시키는 함수
def autoencoder(input_, output_dim, is_training=True, reuse=False, name='auto_anomaly'):
    with tf.variable_scope(name):
        net = layers.fully_connected(input_, 8, activation_fn=None,
                                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        net = tf.nn.relu(net)
        net = layers.fully_connected(net, 4, activation_fn=None,
                                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        net = tf.nn.relu(net)
        net = layers.fully_connected(net, 8, activation_fn=None,
                                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        net = tf.nn.relu(net)
        net = layers.fully_connected(net, output_dim, activation_fn=None,
                                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        net = tf.nn.sigmoid(net)
        
        return net

# Loss값을 계산하기 위한 함수
def loss_fn_sigmoid(logits, label):
    loss = tf.reduce_mean(tf.square(logits - label))

    return loss