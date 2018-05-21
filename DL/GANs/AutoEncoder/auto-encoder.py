import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data_set/MNIST_data/',one_hot = True)

batch_size = 64
train_num = 1000
epoch = 200

def vis_img(samples,epoch):
    fig, axes = plt.subplots(figsize=(7,7),nrows=8, ncols=8, sharey=True, sharex=True)

    #
    for ax, img in zip(axes.flatten(), samples):
        # img = np.array(img.reshape(28,28,1))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((28, 28)),cmap='Greys_r')

    image = 'img/'+str(epoch)+'.jpg'
    plt.savefig(image)
    # plt.show()

    return fig, axes




def encoder(inputs):
    e_layer1 = tf.layers.dense(inputs=inputs,units=256,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                               bias_initializer=tf.constant_initializer(0.1),name='e_layer1')
    # fc1_weights = tf.get_variable('weight', [784, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
    # fc1_biases = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.1))
    # e_layer1 = tf.nn.relu(tf.matmul(inputs, fc1_weights) + fc1_biases)

    e_layer2 = tf.layers.dense(inputs=e_layer1,units=128,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                               bias_initializer=tf.constant_initializer(0.1),name='e_layer2')
    # fc2_weights = tf.get_variable('weight2', [256, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
    # fc2_biases = tf.get_variable('bias2', [128], initializer=tf.constant_initializer(0.1))
    # e_layer2 = tf.nn.relu(tf.matmul(e_layer1, fc2_weights) + fc2_biases)

    return tf.nn.sigmoid(e_layer2)

def decoder(inputs):
    d_layer1 = tf.layers.dense(inputs=inputs,units=256,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                               bias_initializer=tf.constant_initializer(0.1),name='d_layer1')
    # fc1_weights = tf.get_variable('d_weight', [128, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
    # fc1_biases = tf.get_variable('d_bias', [256], initializer=tf.constant_initializer(0.1))
    # d_layer1 = tf.nn.relu(tf.matmul(inputs, fc1_weights) + fc1_biases)

    d_layer2 = tf.layers.dense(inputs=d_layer1,units=784,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),#注意units是全连接层的点数，而非全部点数
                               bias_initializer=tf.constant_initializer(0.1),name='d_layer2')
    # fc2_weights = tf.get_variable('d_weight2', [256, 784], initializer=tf.truncated_normal_initializer(stddev=0.1))
    # fc2_biases = tf.get_variable('d_bias2', [784], initializer=tf.constant_initializer(0.1))
    # d_layer2 = tf.nn.relu(tf.matmul(d_layer1, fc2_weights) + fc2_biases)
    return tf.nn.sigmoid(d_layer2)

x = tf.placeholder(dtype=tf.float32,shape=[None,784],name='input')
# y = tf.placeholder(dtype=tf.float32,shape=[None,128],name='label')

enc = encoder(x)
dec = decoder(enc)

loss = tf.reduce_mean(tf.pow(x - dec,2))
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

all_loss = []
print(mnist.train.num_examples)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epoch):
        for step in range(int(mnist.train.num_examples / batch_size)):
            xs,ys = mnist.train.next_batch(batch_size)
            xs = xs.reshape(-1,784)

            _,losses,out = sess.run([train_op,loss,dec],feed_dict={x:xs})

            if step%100==0:
                print("epoche is %d ---  step is %d   losses is %f"%(e,step,losses))
                all_loss.append(losses)

        out = ((out - out.min()) * 255 / (out.max() - out.min())).astype(np.uint8)
        vis_img(out, e)

    plt.figure()
    plt.plot(all_loss)
    img_name1 = '1.jpg'
    # plt.show()
    plt.savefig(img_name1)
