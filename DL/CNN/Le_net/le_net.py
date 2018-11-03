'''
this is bases on Le-Net-5,change the net constructure , it has 9 layers, 4 conv layers, 2 maxpooling layers, 3 fc layers
use two 3*3 filters replace one 5*5 layers , finally result a little better than original net
'''


import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./src/MNIST_data/",one_hot = True)

def conv(input_tensor,filter,strides,name,padding = "SAME"):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights',filter,initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases',filter[3],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,weights,strides=strides,padding=padding)
        return tf.nn.relu(tf.nn.bias_add(conv1,biases))

def max_pooling(input_tensor,filters,strides,name,padding = "SAME"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_tensor,ksize=filters,strides=strides,padding=padding)


def inference(input_tensor,train,regularizer):

    conv1 = conv(input_tensor,[3, 3, 1, 6],[1,1,1,1],'layer1-conv1',"VALID")

    conv2 = conv(conv1, [3, 3, 6, 12], [1, 1, 1, 1], 'layer1-conv2', "VALID")

    pool1 = max_pooling(conv2,[1,2,2,1],[1,2,2,1],name='pool1',padding="SAME")

    conv3 = conv(pool1,[3,3,12,24],[1,1,1,1],'layer3-conv',padding="VALID")

    conv4 = conv(conv3, [3, 3, 24, 24], [1, 1, 1, 1], 'layer4-conv', padding="VALID")


    pool2 = max_pooling(conv4,[1,2,2,1],[1,2,2,1],name='poo2',padding="SAME")


    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])


    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, 120], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [120], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [120, 84], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [84], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable('weight', [84, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias', [10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases
    return logit



x = tf.placeholder(tf.float32,[None,28,28,1],name='x')
y_ = tf.placeholder(tf.int32,[None],name='y_')

regularizer = tf.contrib.layers.l2_regularizer(0.001)
y = inference(x,False,regularizer)


cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

correct_prediction = tf.equal(tf.cast(tf.argmax(y,1),tf.int32),y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()#use saver save the model
model_path = "model/"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_num = 80000
    batch_size = 128

    x_t = mnist.test.images
    x_t = x_t.reshape(-1,28,28,1)
    y_t = np.argmax(mnist.test.labels,axis = 1)
    test_feed = {x:x_t,y_:y_t}


    x_v = mnist.validation.images
    x_v = x_v.reshape(-1,28,28,1)
    y_v = np.argmax(mnist.validation.labels,axis=1)
    validate_feed = {x:x_v,y_:y_v}

    # #restore the model data
    # ckpt = tf.train.get_checkpoint_state("path")
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, "model/model.ckpt")#reload the saved model

    for i in range(train_num):
        xs,ys = mnist.train.next_batch(batch_size)
        xs = xs.reshape(batch_size,28,28,1)
        ys = np.argmax(ys,axis = 1)

        _,losses = sess.run([train_op,loss],feed_dict={x:xs,y_:ys})

        if i%1000 == 0:
            #saver.save(sess, "model/model.cptk")  # save the train model
            saver.save(sess,os.path.join(model_path,"model.ckpt"),global_step=1000)#save the model
            vali_acc = sess.run(accuracy,feed_dict=validate_feed)
            test_acc = sess.run(accuracy,feed_dict=test_feed)
            print(losses,vali_acc,test_acc)


