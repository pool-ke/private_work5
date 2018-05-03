#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:38:51 2017

@author: root
"""

import numpy as np
from PIL import Image
import skimage.morphology as sm
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import filters,io, measure, color
import os
import math
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class AutoEncoder():
    def __init__(self,H_img,W_img):
        inputs_ = tf.placeholder(tf.float32,(None, H_img, W_img, 1), name = 'inputs_')
        targets_ = tf.placeholder(tf.float32,(None, H_img, W_img, 1), name = 'targets_')
        #hidden layer
        
        conv1_d = tf.layers.conv2d(inputs_,32, (3,3), padding = 'same', activation = tf.nn.relu, name = 'conv1_d')
        conv1_p = tf.layers.max_pooling2d(conv1_d, (2,2),(2,2), padding = 'same', name = 'conv1_p')
        
        conv2_d = tf.layers.conv2d(conv1_p,32, (3,3), padding = 'same', activation = tf.nn.relu, name = 'conv2_d')
        conv2_p = tf.layers.max_pooling2d(conv2_d, (2,2),(2,2), padding = 'same', name = 'conv2_p')
        
        conv3_d = tf.layers.conv2d(conv2_p,32, (3,3), padding = 'same', activation = tf.nn.relu, name = 'conv3_d')
        conv3_p = tf.layers.max_pooling2d(conv3_d, (2,2),(2,2), padding = 'same', name = 'conv3_p')
        
        conv_3_d = tf.layers.conv2d(conv3_p,32, (3,3), padding = 'same', activation = tf.nn.relu, name = 'conv_3_d')
        conv_3_p = tf.layers.max_pooling2d(conv_3_d, (2,2),(2,2), padding = 'same', name = 'conv_3_p')
        
        full_H = math.ceil(H_img/16.0)
        full_W = math.ceil(W_img/16.0)
        
        in_full_connect = tf.reshape(conv_3_p,[-1,full_W*full_H*32], name = 'in_full_connect')#upfold the tensor
        full_connect = tf.layers.dense(in_full_connect, 50, activation = tf.nn.relu, name = 'full_connect')# connect with full
        
        #decoder layer
        
        de_full_connect = tf.layers.dense(full_connect, full_W*full_H*32, activation = tf.nn.relu, name = 'de_full_connect')#connect with full
        de_full = tf.reshape(de_full_connect,[-1,full_H,full_W,32], name = 'de_full')# huifu to the same shape of tensor
        
        conv_4_n = tf.image.resize_nearest_neighbor(de_full,(2*full_H,2*full_W), name = 'conv_4_n')
        conv_4_d = tf.layers.conv2d(conv_4_n, 32, (3,3),padding = 'same',activation = tf.nn.relu, name = 'conv_4_d')
        
        conv4_n = tf.image.resize_nearest_neighbor(conv_4_d,(4*full_H,4*full_W), name = 'conv4_n')
        conv4_d = tf.layers.conv2d(conv4_n, 32, (3,3),padding = 'same',activation = tf.nn.relu, name = 'conv4_d')
        
        conv5_n = tf.image.resize_nearest_neighbor(conv4_d,(8*full_H,8*full_W), name = 'conv5_n')
        conv5_d = tf.layers.conv2d(conv5_n, 32, (3,3),padding = 'same',activation = tf.nn.relu, name = 'conv5_d')
        
        conv6_n = tf.image.resize_nearest_neighbor(conv5_d,(H_img, W_img), name = 'conv6_n')
        conv6_d = tf.layers.conv2d(conv6_n,32, (3,3),padding = 'same',activation = tf.nn.relu, name = 'conv6_d')
        
        logits_ = tf.layers.conv2d(conv6_d, 1, (3,3), padding = 'same', activation = None, name = 'logits_')
        outputs_ = tf.nn.sigmoid(logits_, name = 'outputs_')
        
        #loss function
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = targets_, logits = logits_, name = 'loss')
        cost = tf.reduce_mean(loss, name = 'cost')
        
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        
        self.H_img=H_img
        self.W_img=W_img
        self.inputs_=inputs_
        self.targets_=targets_
        self.outputs_=outputs_
        self.cost=cost
        self.optimizer=optimizer
    
    def read_and_decode(self,filename, H_img, W_img, flag):  #imread  Logo_train.tfrecords
        filename_queue = tf.train.string_input_producer([filename]) #produce a queue 
        reader = tf.TFRecordReader()
        _,serialized_example = reader.read(filename_queue)#return the file name and file
        features = tf.parse_single_example(serialized_example,
                                           features = {
                                                   'label': tf.FixedLenFeature([], tf.int64),
                                                   'img_raw': tf.FixedLenFeature([], tf.string),
                                                   
                                                   })#get the iamge data and label
        img =tf.decode_raw(features['img_raw'], tf.uint8)
        
        img = tf.reshape(img, [H_img,W_img,1]) # reshape an image to size
        #img = tf.cast(img, tf.float32) * (1./255) - 0.5 #give the tensor of image
        label = tf.cast(features['label'], tf.int32)
        if flag == 1:
        
            images, labels = tf.train.shuffle_batch([img, label],batch_size=20,
                                                    capacity = 8000,
                                                    num_threads = 1,
                                                    min_after_dequeue = 2000)
        elif flag == 2:
            images, labels = tf.train.batch([img, label],batch_size = 50,
                                                    capacity = 8000)
        elif flag == 3:
            images, labels = tf.train.shuffle_batch([img, label],batch_size = 50,
                                                    capacity = 8000,
                                                    num_threads = 1,
                                                    min_after_dequeue = 2000)
        elif flag == 4:
             images, labels = tf.train.batch([img, label], batch_size = 50, 
                                            capacity = 8000)
                
        return  images, labels
    
    def train_model(self,path_train):
#    W_img = 695
#    H_img = 161
        epochs = 200
        img, label = self.read_and_decode(path_train, self.H_img, self.W_img,flag = 1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            for i in range(epochs):
              
                val, _ = sess.run([img, label])
                
                val = val/255.0
        
                if i % 10 ==0:
                    saver.save(sess,"QTGUI_Model/model.ckpt")
                    print("model saved")
                batch_cost, _ = sess.run([self.cost, self.optimizer], feed_dict = {self.inputs_: val,self.targets_: val})
                print("Epoch: {}/{}".format(i+1,epochs),
                  "Training loss: {:.4f}".format(batch_cost))
            coord.request_stop()
            coord.join(threads) 
    def test_model(self,image_input):
    #    img2, label2 = read_and_decode("/home/huawei/myfile/code_python/Feng/LOGO_train_More_15.tfrecords",flag = 4)#/home/huawei/myfile/code_python/Feng/tensorflow/LOGO_train_NG_695.tfrecords
        saver2 = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    #        val2, _ = sess.run([img2, label2])
    #        val2 = val2/255.0
            saver2.restore(sess,"./QTGUI_Model/model.ckpt")#./Model/model.ckpt
        
            outp = sess.run(self.outputs_, feed_dict = {self.inputs_: image_input})
            coord.request_stop()
            coord.join(threads)
            return outp