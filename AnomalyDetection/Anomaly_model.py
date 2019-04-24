from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from Anomaly_opsy import *

import numpy as np
import csv
import tensorflow as tf
import tensorflow.contrib.layers as layers
import random

import pandas as pd
import itertools
import pickle
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# 읽어온 Data를 가공하는 class
class c_entity():
    def __init__(self, ent):
        self.features = ent[:-1].astype(float)
        self.label = int(ent[-1])
        

class Anomaly(object):
    # Class 초기화
    def __init__(self, sess, input_size=30, input_size_con=30, entity_size=30, input_width=None, crop=True,
        batch_size=64,  c_dim=1, primary_dim=10, digit_dim=16, reg_para = 0.0005, n_conv=128,
        n_primary=32, n_digit=4, recon_h1=64, recon_h2=128, dataset_name='mnist', checkpoint_dir=None):

        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size

        self.input_size = input_size
        self.input_size_con = input_size_con
        self.entity_size = entity_size

        self.c_dim = c_dim

        self.primary_dim = primary_dim
        self.digit_dim = digit_dim

        self.n_conv = n_conv
        self.n_primary=n_primary
        self.n_digit = n_digit

        self.recon_h1 = recon_h1
        self.recon_h2 = recon_h2
    
        self.recon_output = self.input_size

        self.reg_para = reg_para

        self.dataset_name = dataset_name
        
        self.load_input_anomaly()
        self.build_anomaly()
        
    # Tensor를 build하는 메서드
    def build_anomaly(self):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size), name='inputs')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=(None), name='labels')
        self.is_training = tf.placeholder(tf.bool)
        
        self.anomaly = autoencoder(self.input_x, self.n_digit, self.is_training)
        
        with tf.variable_scope("prediction") as scope:
            self.prob = tf.nn.softmax(self.anomaly)
            
            self.pred_label = tf.argmax(self.prob, axis=1)
            
        with tf.variable_scope("loss") as scope:
            self.loss = loss_fn_sigmoid(self.anomaly, self.input_x)
            
            self.acc = self.accuracy(self.input_y, self.prob)
            
        self.counter = tf.Variable(0, name='global_step', trainable=False)
        

    # Build된 tensor를 사용하여 training하는 메서드
    def train(self, restore=0):
#         opt = layers.optimize_loss(loss=self.loss,
#                                 global_step=self.counter,
#                                 learning_rate=1e-3,
#                                 summaries = None,
#                                 optimizer = tf.train.AdamOptimizer,
#                                 clip_gradients = 0.1)
        optimizer = tf.train.RMSPropOptimizer(1e-5)
        opt = optimizer.minimize(self.loss)

        if restore == 0:
            tf.global_variables_initializer().run()
        else:
            self.restore()

        self.summary_op = tf.summary.merge_all()

        batch_num = int(len(self.train_data)/self.batch_size)

        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

        for epoch in range(100):
            seed = 100
            np.random.seed(seed)
            np.random.shuffle(self.train_data)

            for idx in range(batch_num-1):
                start_time = time.time()
                
                batch = self.train_data[idx*self.batch_size: (idx+1)*self.batch_size]
                batch_x = [data.features for data in batch]
                batch_y = [data.label for data in batch]

                feed_dict = {self.input_x: batch_x, self.input_y: batch_y, self.is_training: True}

                _, loss =  self.sess.run([opt, self.loss], feed_dict=feed_dict)
                total_count = tf.train.global_step(self.sess, self.counter)

                if idx % 100 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, train_loss: %.8f" \
                    % (epoch, idx, batch_num-1, time.time() - start_time, loss))

                    
    # Validation data를 사용하여 성능을 확인하는 메서드
    def validation_check(self, restore=0):
        if restore == 1:
            self.restore()
        
        val_num = int(len(self.train_val_data)/self.batch_size)
        val_loss, val_accuracy = 0.0, 0.0
        for idx in range(val_num-1):
            valid = self.train_val_data[idx*self.batch_size: (idx+1)*self.batch_size]
            valid_x = [data.features for data in valid]
            valid_y = [data.label for data in valid]
            
            feed_dict = {self.input_x: valid_x, self.input_y: valid_y, self.is_training: False}
            loss = self.sess.run([self.loss], feed_dict=feed_dict)
            val_loss += loss[0]
    
        val_loss /=(val_num-1)
        print("[*] Validation: loss = %.8f"\
          %(val_loss))

    # Accuracy를 확인하는 메서드
    def accuracy(self, y, y_pred):
        #y: true one-hot label
        #y_pred: predicted logit
        y = tf.one_hot(y, depth=self.n_digit)
        correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        return accuracy
        
    # 학습할 data를 불러오는 메서드
    def load_input_anomaly(self):
        with open('df_train_new_0.pickle','rb') as f:
            reader = pickle.load(f)
            table_all = np.array(reader)
            
        data = [c_entity(ent) for ent in table_all]
        random.shuffle(data)
        
        self.train_data = data[:int(len(data) * 0.8)]
        self.train_val_data = data[int(len(data) * 0.8):]
        
        
        with open('df_train_new_all.pickle','rb') as f:
            reader = pickle.load(f)
            table_all = np.array(reader)
            
        data = [c_entity(ent) for ent in table_all]
        random.shuffle(data)
        
        self.val_data = data
        
        print('data_setting_done')
        
    # Tensor graph를 저장하는 메서드
    def save(self, checkpoint_path, name):
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        checkpoint_name_path =os.path.join(checkpoint_path,'%s.ckpt'% name)

        value_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
        saver=tf.train.Saver(value_list)
        saver.save(self.sess, checkpoint_name_path)
        
        print('save done!!')
        
    # 저장된 tensor graph를 불러오는 메서드
    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "logA/Anomaly2.ckpt")
    
        print('restore done!!')






    # 최종결과를 plot하는 메서드
    def plot_confusion_matrix(self, check=1, restore=1):
        # This is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the true classifications for the test-set.

        # Get the confusion matrix using sklearn.
        self.batch_size = 1
        
        predict = []
        
        if restore == 1:
            self.restore()
            
        if check == 0:
            print('Please check == 1')
            
        elif check == 1:
            valid_y = [data.label for data in self.val_data]
            for batch_index in range(len(self.val_data) // self.batch_size):
                batch = self.val_data[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
                valid_x = [data.features for data in batch]

                feed_dict = {self.input_x: valid_x, self.is_training: False}
                test_out = self.sess.run(self.loss, feed_dict=feed_dict)
                
                # Anomaly는 loss의 경계 값을 정해야하기 때문에 아래와같이 경계를 만들어준다.
                if test_out < 0.008:
                    test_out_list = [0]
                elif test_out >= 0.008:
                    test_out_list = [1]
                
                for answer in test_out_list:
                    predict.append(answer)
                    
                temp = batch_index
    
            if temp+1 < len(valid_y):
                batch = self.val_data[(temp + 1) * self.batch_size :]
                valid_x = [data.features for data in batch]

                feed_dict = {self.input_x: valid_x, self.is_training: False}
                test_out = self.sess.run(self.loss, feed_dict=feed_dict)
                
                if test_out < 0.008:
                    test_out_list = [0]
                elif test_out >= 0.008:
                    test_out_list = [1]
                    
                for answer in test_out_list:
                    predict.append(answer)
            
            cm = confusion_matrix(y_true=valid_y,
                                      y_pred=predict)
            f1_score = metrics.f1_score(valid_y, predict, average='weighted')
            print('f1_score: %.8f' % f1_score)
        

        # Print the confusion matrix as text.
        print(cm)

        # Plot the confusion matrix as an image.
        plt.matshow(cm)

        # Make various adjustments to the plot.
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, range(2))
        plt.yticks(tick_marks, range(2))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()


