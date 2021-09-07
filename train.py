#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 03:10:33 2018

@author: mukesh
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse as ap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from scipy.ndimage import rotate

# Parameters
parser = ap.ArgumentParser()
parser.add_argument("--lr",help="initial learning rate for GD")
parser.add_argument("--batch_size",help="training batch size (1 and multiples of 5)")
parser.add_argument("--init",help="initialization method")
parser.add_argument("--save_dir",help="saving dir for weights")
parser.add_argument("--load",help="for loading saved model load = 1, to run and save model load = 0")
args=parser.parse_args()
LEARNING_RATE = float(args.lr) #0.00015
EPOCHS = 5
DROPOUT_CONV = 0.8
patience = 5
DROPOUT_HIDDEN = 0.6
BATCH_SIZE = int(args.batch_size)
TEST_BATCH_SIZE = 10
VALIDATION_SIZE = 5000      # Set to 0 to train on all available data
  
# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Weight initialization (Xavier's init)
def weight_xavier_init(n_inputs, n_outputs, init):
    if(init == 1):
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

# Bias initialization
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 2D convolution
def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

# Max Pooling
def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Serve data by batches
def next_batch(batch_size):    
    global train_images
    global labels_flat
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        labels_flat = labels_flat[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], labels_flat[start:end]

# Convert class labels from scalars to one-hot vectors 
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[[index_offset + labels_dense.ravel()]] = 1
    return labels_one_hot

######################batch norm#####################

#####################################################

'''
Preprocessing for MNIST dataset
'''
# Read MNIST data set (Train data from CSV file)
data = pd.read_csv('train.csv')
data_val = pd.read_csv('val.csv')
# Extracting images and labels from given data
# For images
images_val = data_val.iloc[:,1:785].values
images_val = images_val.astype(np.float)
val_labels_flat = data_val['label'].values.ravel()
images = data.iloc[:,1:785].values
images = images.astype(np.float)
labels_flat = data['label'].values.ravel()
###########DATA AUGMENTATION (10%)
#num = 10
#train_tmp1 = np.zeros((int(len(images)/num),784))
#train_tmp2 = np.zeros((int(len(images)/num),784))
#tmp_labels1 = np.zeros((int(len(images)/num),))
#tmp_labels2 = np.zeros((int(len(images)/num),))
              #####Right rotate by 10 degrees
#for i in range(0,len(images),num):
#    tmp_img = np.reshape(images[i], (28,28))
#    result = rotate(tmp_img,10,reshape=False)
#    train_tmp1[int(i/num)] = np.reshape(result, (1,784))
#    tmp_labels1[int(i/num)] = labels_flat[i]
#              #####Left rotate by 10 degrees
#tmp_img = np.reshape(images[0], (28,28))
#result = rotate(tmp_img,-10,reshape=False)
#train_tmp2[0] = np.reshape(result, (1,784))
#tmp_labels2[0] = labels_flat[0]
#for i in range(10,len(images),num):
#    j = i-3
#    tmp_img = np.reshape(images[j], (28,28))
#    result = rotate(tmp_img,-10,reshape=False)
#    train_tmp2[int(i/num)] = np.reshape(result, (1,784))
#    tmp_labels2[int(i/num)] = labels_flat[j]
#images = (np.vstack((images,train_tmp1,train_tmp2)))
#labels_flat = np.append(labels_flat,tmp_labels1)
#labels_flat = np.append(labels_flat,tmp_labels2)

##### Normalize from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
images_val = np.multiply(images_val, 1.0 / 255.0)
val_image_size = images_val.shape[1]
val_image_width = val_image_height = np.ceil(np.sqrt(val_image_size)).astype(np.uint8)

# For labels
labels_count = np.unique(labels_flat).shape[0]
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)
val_labels_count = np.unique(val_labels_flat).shape[0]
val_labels = dense_to_one_hot(val_labels_flat, val_labels_count)
val_labels = val_labels.astype(np.uint8)

# Split data into training & validation
validation_images = images_val
validation_labels = val_labels

train_images = images
labels_flat = labels

'''
Create model with 2D CNN
'''
# Create Input and Output
cost = tf.placeholder('float')
X = tf.placeholder('float', shape=[None, image_size])       
Y_gt = tf.placeholder('float', shape=[None, labels_count])
drop_conv = tf.placeholder('float')
drop_hidden = tf.placeholder('float')

# Model Parameters
W1 = tf.get_variable("W1", shape=[3, 3, 1, 64], initializer=weight_xavier_init(3*3*1, 64, args.init))
W2 = tf.get_variable("W2", shape=[3, 3, 64, 128], initializer=weight_xavier_init(3*3*64, 128, args.init))
W3 = tf.get_variable("W3", shape=[3, 3,128 , 256], initializer=weight_xavier_init(3*3*128, 256, args.init))
W4 = tf.get_variable("W4", shape=[3, 3, 256, 256], initializer=weight_xavier_init(3*3*256, 256, args.init))

W3_FC1 = tf.get_variable("W3_FC1", shape=[256*4*4, 256], initializer=weight_xavier_init(256*4*4, 256, args.init))
W3_FC2 = tf.get_variable("W3_FC2", shape=[256, 256], initializer=weight_xavier_init(256, 256, args.init))
W4_FC2 = tf.get_variable("W4_FC2", shape=[256, labels_count], initializer=weight_xavier_init(256, labels_count, args.init))

B1 = bias_variable([64])
B2 = bias_variable([128])
B3 = bias_variable([256])
B4 = bias_variable([256])
B3_FC1 = bias_variable([256])
B3_FC2 = bias_variable([256])
B4_FC2 = bias_variable([labels_count])


##### CNN model
# Layer 1 - Conv1 + Pool1
X1 = tf.reshape(X, [-1,image_width , image_height,1])                  
l1_conv = tf.nn.relu6(conv2d(X1, W1) + B1)                               
l1_pool = max_pool_2x2(l1_conv)                                         
#l1_drop = tf.nn.dropout(l1_pool, drop_conv)

# Layer 2 - Conv2 + Pool2 + Drop1
l2_conv = tf.nn.relu6(conv2d(l1_pool, W2)+ B2)                           
l2_pool = max_pool_2x2(l2_conv)
l2_drop = tf.nn.dropout(l2_pool, drop_conv) 

#Layer 3 - Conv3
l3_conv = tf.nn.relu6(conv2d(l2_drop, W3)+ B3)
#l3_drop = tf.nn.dropout(l3_conv, drop_conv)  

#layer 4 - Conv4 + Pool 3 + Drop2
l4_conv = tf.nn.relu6(conv2d(l3_conv, W4)+ B4)    
l4_pool = max_pool_2x2(l4_conv)
l4_drop = tf.nn.dropout(l4_pool, drop_conv) 

# Layer 5 - FC1
l5_flat = tf.reshape(l4_drop, [-1, W3_FC1.get_shape().as_list()[0]])
l5_feed = tf.nn.relu6(tf.matmul(l5_flat, W3_FC1)+ B3_FC1) 
l5_drop = tf.nn.dropout(l5_feed, drop_hidden)

# Layer 6 - FC2 Comment for Batch Normalization
l6_feed = tf.nn.relu6(tf.matmul(l5_drop, W3_FC2)+ B3_FC2) 
l6_drop = tf.nn.dropout(l6_feed, drop_hidden)

######Batch Normalization (Uncomment this part)
#w6_BN = tf.Variable(W3_FC2)
#z6_BN = (tf.matmul(l5_drop,w6_BN))
#batch_mean2, batch_var2 = tf.nn.moments(z6_BN,[0])
#scale2 = tf.Variable(tf.ones([256]))
#beta2 = tf.Variable(tf.zeros([256]))
#BN4 = tf.nn.batch_normalization(z6_BN,batch_mean2,batch_var2,beta2,scale2,1e-3)
#l6_feed = tf.nn.relu6(BN4 + B3_FC2) 
#l6_drop = tf.nn.dropout(l6_feed, drop_hidden)

Y_pred = tf.nn.softmax(tf.matmul(l6_drop, W4_FC2)+ B4_FC2) 

# Cost function and training 
cost = -tf.reduce_sum(Y_gt*tf.log(Y_pred))
regularizer = (tf.nn.l2_loss(W3_FC1) + tf.nn.l2_loss(B3_FC1) +tf.nn.l2_loss(W3_FC2) + tf.nn.l2_loss(B3_FC2)  + tf.nn.l2_loss(W4_FC2) + tf.nn.l2_loss(B4_FC2))
cost += 5e-4 * regularizer

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
#train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9, 1e-08).minimize(cost)
#train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
predict = tf.argmax(Y_pred, 1)
#a = np.zeros((1,7,7,256),np.float32)
#for i in range(0,256,26):
#    a[0,3,3,i] = 1
#maskw = tf.Variable(a,name="maskw")

'''
TensorFlow Session
'''
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# start TensorFlow session
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []

T_loss = []
V_loss = []

DISPLAY_STEP=10
val_loss = 0
temp_t_loss = 0
cnt = 0 ##to denote Epochs
st = 50000
if(int(args.load) == 0):
    for i in range(st):
        #get new batch
        batch_xs, batch_ys = next_batch(BATCH_SIZE)        
    
        if i%DISPLAY_STEP == 0 or (i+1) == st:
            train_accuracy = accuracy.eval(feed_dict={X:batch_xs, 
                                                      Y_gt: batch_ys,
                                                      drop_conv: DROPOUT_CONV, 
                                                      drop_hidden: DROPOUT_HIDDEN
                                                      })       
            if(VALIDATION_SIZE):
                validation_accuracy = accuracy.eval(feed_dict={ X: validation_images[0:BATCH_SIZE], 
                                                                Y_gt: validation_labels[0:BATCH_SIZE],
                                                                drop_conv: DROPOUT_CONV,
                                                                drop_hidden: DROPOUT_HIDDEN
                                                                })                                   
                validation_accuracies.append(validation_accuracy)
            train_accuracies.append(train_accuracy)
        # train on batch
        _,train_loss = sess.run([train_op,cost], feed_dict={X: batch_xs, Y_gt: batch_ys, drop_conv: DROPOUT_CONV,
                                               drop_hidden: DROPOUT_HIDDEN
                                               })
        temp_t_loss += train_loss
        # check final accuracy on validation set  
        if(i%(int(len(images)/BATCH_SIZE)) == 0):
            val_loss_new = sess.run([cost],feed_dict={X: validation_images, 
                                                           Y_gt: validation_labels,
                                                           drop_conv: 1.0, 
                                                           drop_hidden: 1.0
                                                           })
        ####Normalizing losses
            val_loss_new[0] /= len(images_val)
            temp_t_loss /= len(images)
            print("Epoch : ",cnt,", train loss : ", temp_t_loss, ", val loss : ", val_loss_new[0])    
            cnt += 1
            V_loss.append(val_loss_new[0])
            T_loss.append(temp_t_loss)
            temp_t_loss = 0
            validation_accuracy = accuracy.eval(feed_dict={ X: validation_images, 
                                                                Y_gt: validation_labels,
                                                                drop_conv: 1.0,
                                                                drop_hidden: 1.0})     
            
            print(' validation_accuracy => %.4f '%(validation_accuracy))
            #####EARLY STOPPING
            if(validation_accuracy < 0.1043 and validation_accuracy > 0.1041):
                break
            if(val_loss < val_loss_new[0]):
                patience = patience-1
                #LEARNING_RATE *= 0.5
            else:
                patience = 5
                saver = tf.train.Saver()
                saver.save(sess, args.save_dir+'/model.ckpt')
    #            if(val_loss_new[0] < 900):
    #                print("DONE")
    #                break
            if(patience == 0):
                break
            val_loss = val_loss_new[0]
            ####When Epochs complete
            if(cnt > EPOCHS):
                print("Epochs Completed")
                break
#####Importing saved model
else:
    print("Importing saved model")
    saver = tf.train.Saver()
    saver.restore(sess, args.save_dir+'/model.ckpt')
    print("Model imported")

######TESTING
test_images = pd.read_csv('test.csv')
test_images = test_images.iloc[:,1:785].values
test_images = test_images.astype(np.float)
# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)
# Predicting labels
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//TEST_BATCH_SIZE):
    predicted_lables[i*TEST_BATCH_SIZE : (i+1)*TEST_BATCH_SIZE] = predict.eval(feed_dict={X: test_images[i*TEST_BATCH_SIZE : (i+1)*TEST_BATCH_SIZE],drop_conv: 1.0,
                    drop_hidden: 1.0})

# save results
np.savetxt(args.save_dir+'/submission.csv', 
           np.c_[range(0,len(test_images)),predicted_lables], 
           delimiter=',', 
           header = 'id,label', 
           comments = '', 
           fmt='%d')

# save train loss vs val loss values
np.savetxt("plot.csv", np.column_stack((T_loss, V_loss)), delimiter=",")
sess.close()