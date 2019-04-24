from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from Entity_opsy import *

import numpy as np
import csv
import tensorflow as tf
import tensorflow.contrib.layers as layers
import random

import pandas as pd
import itertools
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# 읽어온 Data를 가공하는 class
class c_entity():
    def __init__(self, ent):
        self.cat = ent[:61].astype(int)
        self.con = ent[61:-1].astype(float)
        self.label = int(ent[-1])
        
        self.features = ent[:-1].astype(float)
        

class Entity_embedding(object):
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
        
        self.load_input_entity()
        self.build_entity()
        
    # Tensor를 build하는 메서드
    def build_entity(self):
        self.primary_caps_layer = CapsConv(self.primary_dim, name='primary_caps')
        self.digit_caps_layer= CapsConv(self.digit_dim, name='digit_caps')
        
        self.cat_x = tf.placeholder(dtype=tf.int32, shape=(None, self.input_size), name='inputs')
        self.con_x = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size_con), name='inputs')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=(None), name='labels')
        self.is_training = tf.placeholder(tf.bool)
        
        # Entity가 만들어지는 scope
        with tf.variable_scope("embedding") as scope:
            temp = []
            for i in range(self.input_size):
                temp.append(entity_layer(tf.transpose(self.cat_x)[i], self.entity_size[i]))
            
            self.temp1 = tf.concat(temp, 1)
            self.temp2 = tf.concat([self.temp1, self.con_x], 1) # 만들어진 entity를 활용하기위해 따로 저장
            
            
            temp.append(fc_1layer(self.con_x, 16))
            
            self.emb = tf.concat(temp, 1)
            
        with tf.variable_scope('capsule') as scope:
            self.input_x_expand = tf.expand_dims(self.emb, axis=-1, name='input_expand_1')
            self.first_fc = conv1d(self.input_x_expand, self.input_size, 5, self.is_training)
            self.primary_caps = self.primary_caps_layer(self.first_fc, self.n_digit, is_training=self.is_training)
            self.digit_caps = self.digit_caps_layer(self.primary_caps, self.n_digit, is_training=self.is_training)
            
        with tf.variable_scope("prediction") as scope:
            self.logit = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=-1)) #[batch_size, num_caps]
            self.prob = tf.nn.softmax(self.logit)
            self.pred_label = tf.argmax(self.prob, axis=1)
            
        with tf.variable_scope("loss") as scope:
            self.loss = margin_loss(self.logit, self.input_y, n_digit=self.n_digit)
            
            self.acc = self.accuracy(self.input_y, self.prob)
            
        self.counter = tf.Variable(0, name='global_step', trainable=False)

    # Build된 tensor를 사용하여 training하는 메서드
    def train(self, restore=0):
        opt = layers.optimize_loss(loss=self.loss,
                                global_step=self.counter,
                                learning_rate=1e-3,
                                summaries = None,
                                optimizer = tf.train.AdamOptimizer,
                                clip_gradients = 0.1)

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
                batch_cat = [data.cat for data in batch]
                batch_con = [data.con for data in batch]
                batch_y = [data.label for data in batch]

                feed_dict = {self.cat_x: batch_cat, self.con_x: batch_con, self.input_y: batch_y, self.is_training: True}

                _, loss, train_accuracy, pred_label =  self.sess.run([opt, self.loss, self.acc, self.pred_label], feed_dict=feed_dict)
                total_count = tf.train.global_step(self.sess, self.counter)

                if idx % 10 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, train_loss: %.8f, train_accurracy: %.8f" \
                    % (epoch, idx, batch_num-1, time.time() - start_time, loss, train_accuracy))
                    
                    

    # Validation data를 사용하여 성능을 확인하는 메서드
    def validation_check(self, restore=0):
        if restore == 1:
            self.restore()
        
        val_num = int(len(self.val_data)/self.batch_size)
        val_loss, val_accuracy = 0.0, 0.0
        for idx in range(val_num-1):
            valid = self.val_data[idx*self.batch_size: (idx+1)*self.batch_size]
            valid_cat = [data.cat for data in valid]
            valid_con = [data.con for data in valid]
            valid_y = [data.label for data in valid]
            
            feed_dict = {self.cat_x: valid_cat, self.con_x: valid_con, self.input_y: valid_y, self.is_training: False}
            loss, accuracy = self.sess.run([self.loss, self.acc], feed_dict=feed_dict)
            val_loss += loss
            val_accuracy += accuracy
    
        val_loss /=(val_num-1)
        val_accuracy /= (val_num-1)
        print("[*] Validation: loss = %.8f, accuracy: %.8f"\
          %(val_loss, val_accuracy))

    # Accuracy를 확인하는 메서드
    def accuracy(self, y, y_pred):
        #y: true one-hot label
        #y_pred: predicted logit
        y = tf.one_hot(y, depth=self.n_digit)
        correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        return accuracy
        
    # 학습할 data를 불러오는 메서드
    def load_input_entity(self):
        with open('df_tr_modified.pickle','rb') as f:
            reader = pickle.load(f)
            table_all = np.array(reader)
            
        data = [c_entity(ent) for ent in table_all]
        random.shuffle(data)
        
        self.train_data = data[:int(len(data) * 0.8)]
        self.val_data = data[int(len(data)*0.8):]
        
    # Testing할 data를 불러오는 메서드
    def load_input_test(self):
        with open('df_tr_modified_test.pickle','rb') as f:
            reader = pickle.load(f)
            table_all = np.array(reader)
            
        data = [c_entity(ent) for ent in table_all]
        random.shuffle(data)
        
        self.val_data = data
        
        
        print('data_setting_test_done')
        
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
        saver.restore(self.sess, "Entity/E_embedding3.ckpt")
    
        print('restore done!!')





    
    # 최종결과를 plot하는 메서드
    def plot_confusion_matrix(self, check=0, restore=0):
        # This is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the true classifications for the test-set.

        # Get the confusion matrix using sklearn.
#         self.load_input_test()
        
        predict = []
        
        if restore == 1:
            self.restore()
            
        # Entity를 randomforest에 적용
        if check == 0:
            valid = self.train_data
            valid_cat = [data.cat for data in valid]
            valid_con = [data.con for data in valid]
            valid_y = [data.label for data in valid]
            valid_feature = [data.features for data in valid]
            
            feed_dict = {self.cat_x: valid_cat, self.con_x: valid_con, self.is_training: False}
            test_out = self.sess.run(self.temp2, feed_dict=feed_dict)
            test_out_list = test_out.tolist()
            
            m= RandomForestClassifier(n_estimators=40 , min_samples_leaf=5, max_features=0.4, random_state=22, class_weight={0:1, 1:4})
            m.fit(test_out_list, valid_y)
            
            
            valid = self.val_data
            valid_cat = [data.cat for data in valid]
            valid_con = [data.con for data in valid]
            valid_y = [data.label for data in valid]
            valid_feature = [data.features for data in valid]
            
            feed_dict = {self.cat_x: valid_cat, self.con_x: valid_con, self.is_training: False}
            test_out = self.sess.run(self.temp2, feed_dict=feed_dict)
            test_out_list = test_out.tolist()
            
            y_pred = m.predict(test_out_list)
            
            
            cm = confusion_matrix(y_true=valid_y,
                              y_pred=y_pred)
            
            f1_score = metrics.f1_score(valid_y, y_pred, average='weighted')
            print('f1_score: %.8f' % f1_score)
            
        # Entity를 CapsuleNet에 적용
        elif check == 1:
            valid_y = [data.label for data in self.val_data]
            for batch_index in range(len(self.val_data) // self.batch_size):
                batch = self.val_data[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
                valid_cat = [data.cat for data in batch]
                valid_con = [data.con for data in batch]

                feed_dict = {self.cat_x: valid_cat, self.con_x: valid_con, self.is_training: False}
                test_out = self.sess.run(self.pred_label, feed_dict=feed_dict)
                test_out_list = test_out.tolist()
                for answer in test_out_list:
                    predict.append(answer)
                    
                temp = batch_index
    
            if temp+1 < len(valid_y):
                batch = self.val_data[(temp + 1) * self.batch_size :]
                valid_cat = [data.cat for data in batch]
                valid_con = [data.con for data in batch]

                feed_dict = {self.cat_x: valid_cat, self.con_x: valid_con, self.is_training: False}
                test_out = self.sess.run(self.pred_label, feed_dict=feed_dict)
                test_out_list = test_out.tolist()
                for answer in test_out_list:
                    predict.append(answer)
                    
            cm = confusion_matrix(y_true=valid_y,
                                      y_pred=predict)
            f1_score = metrics.f1_score(valid_y, predict, average='weighted')
            print('f1_score: %.8f' % f1_score)
            
            
        # Randomforest
        elif check == 2:
            valid = self.train_data
            valid_cat = [data.cat for data in valid]
            valid_con = [data.con for data in valid]
            valid_y = [data.label for data in valid]
            valid_feature = [data.features for data in valid]
            
            m= RandomForestClassifier(n_estimators=40 , min_samples_leaf=5, max_features=0.4, random_state=22, class_weight={0:1, 1:4})
            m.fit(valid_feature, valid_y)
            
            
            valid = self.val_data
            valid_cat = [data.cat for data in valid]
            valid_con = [data.con for data in valid]
            valid_y = [data.label for data in valid]
            valid_feature = [data.features for data in valid]
            
            y_pred = m.predict(valid_feature)
            
            
            cm = confusion_matrix(y_true=valid_y,
                              y_pred=y_pred)
            
            f1_score = metrics.f1_score(valid_y, y_pred, average='weighted')
            print('f1_score: %.8f' % f1_score)
        

#         Print the confusion matrix as text.
        print(cm)

#         # Plot the confusion matrix as an image.
        plt.matshow(cm)

#         # Make various adjustments to the plot.
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, range(2))
        plt.yticks(tick_marks, range(2))
        plt.xlabel('Predicted')
        plt.ylabel('True')

#         # Ensure the plot is shown correctly with multiple plots
#         # in a single Notebook cell.
        plt.show()
    
    
    # T-SNE를 사용하여 Entity들의 거리를 확인하는 메서드
    def visualization(self, restore=1):
        if restore == 1:
            self.restore()
        
        valid = self.val_data
        valid_cat = [data.cat for data in valid]
        valid_con = [data.con for data in valid]
        valid_y = [data.label for data in valid]
        valid_feature = [data.features for data in valid]

        
        feed_dict = {self.cat_x: valid_cat, self.con_x: valid_con, self.is_training: False}
        test_out = self.sess.run(self.temp1, feed_dict=feed_dict)
        test_out_list = test_out.tolist()
        
        
        model = TSNE(learning_rate=100)
        transformed = model.fit_transform(test_out_list)

        xs = transformed[:,0]
        ys = transformed[:,1]
        plt.scatter(xs,ys,c=valid_y)

        plt.show()


