import math
import numpy as np 
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers


#shape of vector: [None, ...(# of capsules)..., Caps-D]
def squash(vector,  axis=-1, epsilon=1e-5, name='squash_op'):
    with tf.name_scope(name) as scope:
        vec_norm_sq = tf.reduce_sum(tf.square(vector), axis=axis, keep_dims=True)
        vec_norm = tf.sqrt(vec_norm_sq+epsilon) #prevents from zero division
        scale = (vec_norm_sq)/(1+vec_norm_sq)
        squashed_vector = scale * (vector/vec_norm)
        
        return squashed_vector

# CapsuleNet에서의 margin loss를 계산하는 함수
def margin_loss(logit, pred_label, m_plus=0.9, m_minus=0.1, n_digit=2, lamb=0.5):
    pred_label = tf.one_hot(pred_label, depth=n_digit)
    print(logit)
    print(pred_label)
    print(tf.square(tf.maximum(0., m_plus-logit)))

    loss = pred_label * tf.square(tf.maximum(0., m_plus-logit))
    loss += lamb* (1-pred_label) * tf.square(tf.maximum(0., logit-m_minus))
    
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

# Reconstruction된 data를 원본 data와 비교하여 loss값을 계산하는 함수
def reconstruction_loss(input_, recon):
    input_dim = input_.get_shape().as_list()[1:]
    len_dim = len(input_dim)
    input_size = 1
    for i in range(len_dim): input_size *=input_dim[i]

    input_ = tf.reshape(input_, [-1, input_size])
    recon = tf.reshape(recon, [-1, input_size])

    return tf.reduce_mean(tf.reduce_sum(tf.square(input_ - recon), axis=-1))

    
# Continuous feature를 위한 함수
def fc_1layer(input_, output_dim, is_training=True, reuse=False, name='fc_1layer'):
    with tf.variable_scope(name):
        net = layers.fully_connected(input_, output_dim, activation_fn=None,
                                         weights_initializer=tf.contrib.layers.xavier_initializer())
        
        return net
    
# Primary capsules를 위해 conv1d를 하는 함수
def conv1d(input_, input_size, output_dim, is_training=True, reuse=False, name='conv1d_layer'):
    with tf.variable_scope(name):
        with tf.variable_scope('first_layer'):
            net = tf.layers.conv1d(inputs=input_, filters=256, kernel_size=input_size, activation=None)
            net = layers.batch_norm(net, decay=0.9, updates_collections=None, activation_fn=None,
                                center=True, scale=True, zero_debias_moving_mean=True, is_training=is_training)
            net = tf.nn.relu(net)
        with tf.variable_scope('second_layer'):
            net = tf.layers.conv1d(inputs=net, filters=output_dim * 10, kernel_size=1, activation=None)
            net = layers.batch_norm(net, decay=0.9, updates_collections=None, activation_fn=None,
                                center=True, scale=True, zero_debias_moving_mean=True, is_training=is_training)
            net = tf.nn.relu(net)
        print(net)
        
        return net
    
# Categorical feature를 위한 entity embedding 함수
def entity_layer(input_, entity_size, is_training=True, reuse=False, name='entity_layer'):
    embedding_size = min(np.ceil((entity_size) / 2) + 1, 50)
    
    embeddings = tf.Variable(tf.random_uniform([entity_size, embedding_size.astype(int)], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, input_, partition_strategy='div')
    
    return embed
    

# Reconstruction을 하기위한 함수
def fc_layer(input_, output_dim, initializer = tf.truncated_normal_initializer(stddev=0.02), activation='linear', name=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name or "Linear") as scope:
        if len(shape) > 2 : input_ = tf.layers.flatten(input_)
        w = tf.get_variable("fc_w", [shape[1], output_dim], dtype=tf.float32, initializer = initializer)
        b = tf.get_variable("fc_b", [output_dim], initializer = tf.constant_initializer(0.0))

        result = tf.matmul(input_, w) + b

        if activation == 'linear':
            return result
        elif activation == 'relu':
            return tf.nn.relu(result)
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(result)

# CapsuleNetwork의 계산을 위한 class
class CapsConv(object):
    def __init__(self, d_caps, name=None,  is_train=True):
        self.d_caps = d_caps
        self.name = name if not name == None else "primary_caps"
        self.reuse = is_train

    def __call__(self, input_, n_caps, is_training, kernel_size=9, stride_size=2, route_iter=3, initializer = tf.truncated_normal_initializer(stddev=0.02)):
        input_shape = input_.get_shape().as_list()
        batch_size = tf.shape(input_)[0]
        #[None, height, width, channel]
    
        if 'primary' in self.name: #Primary Capsule
            with tf.variable_scope(self.name) as scope:
                    
                print((input_shape[1]*input_shape[2])/self.d_caps)
                total_num_cap = int((input_shape[1]*input_shape[2])/self.d_caps)

                capsules = tf.reshape(input_, [batch_size, total_num_cap, self.d_caps])
                print('--capsules--')
                print(capsules)
                capsules = squash(capsules)
                print(capsules)
                
                return capsules

        else: #Digit Capsule
            with tf.variable_scope(self.name) as scope:
                #let assume input_: [batch_size, # of capsules, d-capsules]
                #if len(input_shape) ==2 : input_ = tf.expand_dims(input_, axis=-1)
                self.input_shape = input_.get_shape().as_list()
                input_tiled = tf.expand_dims(input_, axis=-1, name='input_expand_1')#[batch_size, # of capsule, ..., d-capsules, 1]
                input_tiled = tf.expand_dims(input_tiled, axis=2, name='input_expand_2')#[batch_size, # of capsule, ..., d-capsules, 1]
                print('--input_tiled--')
                print(input_tiled)
                
                input_tiled = tf.tile(input_tiled, [1,1, n_caps, 1,1], name='input_tile')#[batch_size, # of capsule, # of next capsule, d_capsules, 1]
                print('--input_tiled--')
                print(input_tiled)

                W = tf.get_variable('prediction_w', [1,self.input_shape[1], n_caps, self.d_caps, self.input_shape[2]], initializer = initializer)
                W_tiled = tf.tile(W, [batch_size, 1,1,1,1], name = 'W_tiled')

                prediction_vectors = tf.matmul(W_tiled, input_tiled)

                b = tf.zeros([batch_size, self.input_shape[1], n_caps,1,1])

                for i in range(route_iter):
                    coupling_coeff = tf.nn.softmax(b,dim=2)

                    s = tf.multiply(prediction_vectors, coupling_coeff,name='weighted_prediction')
                    sum_s = tf.reduce_sum(s, axis=1, keep_dims=True, name='weighted_sum')
                    capsules = squash(sum_s, axis=-2) #(None, 1, # of nex capsule capsule, d_capsule, 1)
                    caps_out_tile = tf.tile(capsules, [1, self.input_shape[1], 1,1,1], name='capsule_output_tiled')

                    a = tf.matmul(prediction_vectors, caps_out_tile, transpose_a=True, name='agreement')
                    b = tf.add(b, a, name='update_logit')

                capsules = tf.reshape(capsules, [batch_size, n_caps, self.d_caps])
                print('--capsules--')
                print(capsules)
                print('--capsules--')
                #return tf.squeeze(capsules)
                return capsules