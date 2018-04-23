########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names

import sys

import tensorlayer as tl
from utils1 import *

import csv
import random as rd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class vgg16:
    # weights 就是代指 'vgg16_weights.npz'， imgs是feed了image的tensor
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        #conv+maxpool
        self.convlayers()
        #fc
        self.fc_layers()

        #only run this one when final testing????
        #final score, softmax
        #self.probs = tf.nn.softmax(self.fc3l)

        # 10-9
        #self.probs = tf.nn.softmax(self.fc1)
        #self.fc1Val = self.fc1
        self.fc1Val = self.fc1


        if weights is not None and sess is not None:
            #输出检查load进去weight数目合不合理,以及assign所有weight进net
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        with tf.name_scope('preprocess') as scope:
            
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean  #??????
            
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
           
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            # 合并w，b
            out = tf.nn.bias_add(conv, biases)
            # 调整w，b
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
 
        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')

        # After pool5, it becomes 7*7*512 nueurons. (224*1/32（5 times maxpool reducing spatial） = 7; Depth is 512)
        
    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            print("shape.fc1", shape)
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print (i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

def read_all_imgs(img_list, path='', n_threads=1, dataset = ''):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn1, path=path, dataset = dataset)
        imgs.extend(b_imgs)
        print('read %d from %s belongs to %s' % (len(imgs), path, dataset))
    return imgs

def train(data, i):

    if i == 0:
        hr_img_path = './image/Fabric/'
    if i == 1:
        #hr_img_path = './image/Foliage/'
        hr_img_path = './image/Leather/'
    if i == 2:
        hr_img_path = './image/Fur/'
    if i == 3:
        hr_img_path = './image/Glass/'
    if i == 4:
        hr_img_path = './image/Leather/'
    if i == 5:
        hr_img_path = './image/Metal/'
    if i == 6:
        hr_img_path = './image/Paper/'
    if i == 7:
        hr_img_path = './image/Plastic/'
    if i == 8:
        hr_img_path = './image/Sky/'
    if i == 9:
        hr_img_path = './image/Stone/'
    if i == 10:
        hr_img_path = './image/Water/'
    if i == 11:
        hr_img_path = './image/Wood/'
    # Fabric Foliage Fur Glass Leather Metal Paper Plastic Sky Stone Water Wood
    dataset = 'train'

    train_all_list = sorted(tl.files.load_file_list(path=hr_img_path, regx='.*.png', printable=False))
    train_num = 70
    train_hr_img_list = sorted(rd.sample(tl.files.load_file_list(path=hr_img_path, regx='.*.png', printable=False), train_num))
    
    print ("train_hr_img_list[:]", train_hr_img_list)
    train_hr_imgs = read_all_imgs(train_hr_img_list[0:data], path=hr_img_path, n_threads=1, dataset = dataset)
    test_hr_img_list = sorted([x for x in train_all_list if x not in train_hr_img_list])
    print ("test_hr_img_list[:]", test_hr_img_list)
    
    return train_hr_imgs, test_hr_img_list

def test(data, i, test_list):

    if i == 0:
        hr_img_path = './image/Fabric/'
    if i == 1:
        #hr_img_path = './image/Foliage/'
        hr_img_path = './image/Leather/'
    if i == 2:
        hr_img_path = './image/Fur/'
    if i == 3:
        hr_img_path = './image/Glass/'
    if i == 4:
        hr_img_path = './image/Leather/'
    if i == 5:
        hr_img_path = './image/Metal/'
    if i == 6:
        hr_img_path = './image/Paper/'
    if i == 7:
        hr_img_path = './image/Plastic/'
    if i == 8:
        hr_img_path = './image/Sky/'
    if i == 9:
        hr_img_path = './image/Stone/'
    if i == 10:
        hr_img_path = './image/Water/'
    if i == 11:
        hr_img_path = './image/Wood/'
    # Fabric Foliage Fur Glass Leather Metal Paper Plastic Sky Stone Water Wood
    dataset = 'test'

    #train_hr_img_list = sorted(tl.files.load_file_list(path=hr_img_path, regx='.*.png', printable=False))
    #print ("train_hr_img_list[:]", train_hr_img_list)
    train_hr_imgs = read_all_imgs(test_list[0:data], path=hr_img_path, n_threads=1, dataset = dataset)   
    
    return train_hr_imgs



if __name__ == '__main__':
    #tf.Session.run also optionally takes a dictionary of feeds, which is a mapping from tf.Tensor objects (typically tf.placeholder tensors) to values
    # (typically Python scalars, lists, or NumPy arrays) that will be substituted for those tensors in the execution.
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    #data = 67
    #data = 33
    row_cross = []
    X = []
    X_test = []

    #with open('train_4corF1.csv', 'w', newline='') as csvfile1:
        #with open('test_4corF1.csv', 'w', newline='') as csvfile2:
    with open('train_patch32.csv', 'w', newline='') as csvfile1:
        with open('test_patch32.csv', 'w', newline='') as csvfile2:

            spamwriter1 = csv.writer(csvfile1, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter2 = csv.writer(csvfile2, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(3):
                img1, test_list = train(70, i)
                img2 = test(30, i, test_list)
                for i in range(70):
                    print ("i", i)
                    # 4 imgs of corners / each LF extract how many subviews
                    for j in range (5) :
                        prob = sess.run(vgg.fc1Val, feed_dict={vgg.imgs: [img1[i][j]]})[0]
                        #print (np.array(prob).shape)
                        #prob = np.reshape(prob, (1, 14*14*512))
                        #print (np.array(prob).shape)
                        #row_cross = row_cross + list(prob[0])
                        #print (np.array(row_cross).shape)
                        #sys.exit("==========done writing csv============")
                        #row_cross = row_cross + list(prob)
                        spamwriter1.writerow(prob)
                    #spamwriter1.writerow(row_cross)
                    #X.append(row_cross)
                    print (np.array(prob).shape)
                    row_cross = []
                for i in range(30):
                    print ("i", i)
                    # 4 imgs of corners
                    for j in range (5) :
                        prob = sess.run(vgg.fc1Val, feed_dict={vgg.imgs: [img2[i][j]]})[0]
                        #prob = np.reshape(prob, (1, 14*14*512))
                        #row_cross = row_cross + list(prob[0])
                        #row_cross = row_cross + list(prob)
                        spamwriter2.writerow(prob)
                    #spamwriter2.writerow(row_cross)
                    #X_test.append(row_cross)
                    print (np.array(prob).shape)
                    row_cross = []

    '''
    sys.exit("==========done writing csv============")
    y = [0]*67
    for i in range (67):
        y.append(1)
    for i in range (67):
        y.append(2)
    for i in range (67):
        y.append(3)
    for i in range (67):
        y.append(4)
    for i in range (67):
        y.append(5)
    for i in range (67):
        y.append(6)
    for i in range (67):
        y.append(7)
    for i in range (67):
        y.append(8)
    for i in range (67):
        y.append(9)
    for i in range (67):
        y.append(10)
    for i in range (67):
        y.append(11)
    print (len(y))
    print (set(y))


    y_ground = [0]*33
    for i in range (33):
        y_ground.append(1)
    for i in range (33):
        y_ground.append(2)
    for i in range (33):
        y_ground.append(3)
    for i in range (33):
        y_ground.append(4)
    for i in range (33):
        y_ground.append(5)
    for i in range (33):
        y_ground.append(6)
    for i in range (33):
        y_ground.append(7)
    for i in range (33):
        y_ground.append(8)
    for i in range (33):
        y_ground.append(9)
    for i in range (33):
        y_ground.append(10)
    for i in range (33):
        y_ground.append(11)
    print (len(y_ground))
    print (set(y_ground))

    logistic = LogisticRegression()
    logistic.fit(X,y)

    y_pred = logistic.predict(X_test)
    print ("y_pred", set(y_pred))

    if (len(y_ground) == len(y_pred)):
        print ("ready to predict")
        print (accuracy_score(y_ground, y_pred))

    my_dict = {1:'Fabric', 2:'Foliage', 3:'Fur', 4:'Glass', 5:'Leather', 6:'Metal', 7:'Paper', 8:'Plastic', 9:'Sky', 10:'Stone', 11:'Water', 12:'Wood'}
    for i in range(12):
        print(my_dict[i+1], accuracy_score(y_ground[(i*33):(i+1)*33], y_pred[(i*33):(i+1)*33]))

    '''
