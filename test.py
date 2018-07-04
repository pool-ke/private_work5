# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
path = "./train/"
# 创建一个changeImage方法将base64图片转化为模型可用的数据集，方便服务器之间传输
def readImage(path, patch_size):
    classes = {"pos":1, "neg":0}
    pos_data = []
    neg_data = []
    neg_y = []
    pos_y = []
    for cl in classes.keys():
        base_dir = path + cl + "/"
        files = os.listdir(base_dir)
        for f in files:
            img = cv.imread(base_dir + f, 0)

            if img.shape[0] !=patch_size or img.shape[1] != patch_size:
                img = cv.resize(img, (patch_size, patch_size))
            image_data = img.reshape((-1,patch_size*patch_size))
            image_data = image_data.tolist()[0]
            if(cl == 'neg'):
                neg_data.append(image_data)
                temp = [1, 0]
                neg_y.append(temp)
            if(cl == 'pos'):
                pos_data.append(image_data)
                temp = [0, 1]
                pos_y.append(temp)
    return pos_data, neg_data, neg_y, pos_y

learning_rate = 0.001 #学习率 

patch_size = 28
num_steps = 300 #
display_step = 10 #显示间隔
save_step = 50
num_input = patch_size*patch_size #image shape:28*28
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.75 #用于随机丢弃，防止过拟

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

#创建模型
def conv_net(x, weights, biases, dropout):
    patch_size = 28

    x = tf.reshape(x, shape=[-1, patch_size, patch_size, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# 设置权重和偏移
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([int(patch_size/4)*int(patch_size/4)*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)
pred = tf.argmax(prediction, 1)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(pred, tf.argmax(Y, 1))
relabel = tf.argmax(Y, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.trainable_variables())

def test(pos_data, neg_data, neg_y, pos_y):
    #neg = np.random.randint(len(neg_data),size=len(neg_data))
    neg_data1 = np.array(neg_data).tolist()
    batch_x = pos_data + neg_data1
    batch_y = pos_y + neg_y[0:len(neg_data)]
    sess.run(init)
    saver.restore(sess, "./model/model.ckpt")#./Model/model.ckpt
    print(sess.run(accuracy, feed_dict={X: batch_x,
                                      Y: batch_y,
                                      keep_prob: 1.0}))
    pre, label = sess.run([pred, relabel], feed_dict={X: batch_x,
                                      Y: batch_y,
                                      keep_prob: 1.0})
#    relabel = np.array(relabel)
    
    TN_FP = pre[label < 1]
    FP = TN_FP[TN_FP > 0]
    FPR = FP.size/TN_FP.size
    print("FPR", FPR)
    
    TP_FN = pre[label > 0]
    TP = TP_FN[TP_FN > 0]
    TPR = len(TP)/len(TP_FN)
    print("TPR", TPR)


test_path = './test1/28/'
pos_data, neg_data, neg_y, pos_y  = readImage(test_path, patch_size )

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
with tf.Session() as sess:
    test(pos_data, neg_data, neg_y, pos_y )
