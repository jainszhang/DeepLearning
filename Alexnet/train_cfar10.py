
import os

import numpy as np
import tensorflow as tf
import random
import Alexnet
from 经典网络 import read_data
# from datagenerator import ImageDataGenerator
from datetime import datetime
# from tensorflow.contrib.data import Iterator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_file = 'train.txt'
val_file = 'val.txt'

cifar10_dir = '../data_set/cifar-10-batches-py/'

# Learning params
learning_rate = 0.001
num_epochs = 1
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 10



display_step = 20

filewriter_path = "tensorboard"
checkpoint_path = "model/"

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


model_path = "model/"




#申请位置
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)#dropout rate

model = Alexnet.AlexNet(x, keep_prob, num_classes)
score = model.fc8


with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y_))

train_op = tf.train.AdamOptimizer(0.00002).minimize(loss)

with tf.name_scope("accuracy"):
    # correct_prediction = tf.equal(tf.cast(tf.argmax(score,1)),y_)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


X_train, Y_train, X_test, Y_test  = read_data.load_CIFAR10(cifar10_dir)

Y_train_labels = np.zeros((len(Y_train),10),np.uint8)
Y_test_labels = np.zeros((len(Y_test),10),np.uint8)
for i in range(len(Y_train)):
    Y_train_labels[i][Y_train[i]] = 1

for i in range(len(Y_test)):
    Y_test_labels[i][Y_test[i]] = 1



def getBatch(begin):
    batch_x_train = X_train[begin*batch_size:begin*batch_size+batch_size]
    batch_y_train = Y_train_labels[begin*batch_size:begin*batch_size+batch_size]
    return batch_x_train,batch_y_train

def getBatch1(begin):
    test_x = X_test[begin*200:(begin+1)*200]
    test_y = Y_test_labels[begin*200:(begin+1)*200]
    return test_x,test_y

train_batches_per_epoch = int(np.floor(X_train.shape[0]/batch_size))
#val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

saver = tf.train.Saver()#use saver save the model
model_path = "model/"

with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())#初始化所有变量

    # Loop over number of epochs
    for epoch in range(num_epochs):#多少个epoch，迭代多少次
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        for step in range(train_batches_per_epoch):
            x_s, y_s = getBatch(step)
            k = 1
            _,losses = sess.run([train_op,loss], feed_dict={x: x_s,y_: y_s,
                                          keep_prob: dropout_rate})

            if step % 10 == 0:
                saver.save(sess,os.path.join(model_path,"model.ckpt"),global_step=10)#save the model

            print("the %d step's loss is %f"%(step,losses))















    #
    #         # Generate summary with the current batch of data and write to file
    #         if step % display_step == 0:
    #             s = sess.run(merged_summary, feed_dict={x: img_batch,
    #                                                     y: label_batch,
    #                                                     keep_prob: 1.})
    #
    #             writer.add_summary(s, epoch*train_batches_per_epoch + step)
    #
    #     # Validate the model on the entire validation set
    #     print("{} Start validation".format(datetime.now()))
    #     sess.run(validation_init_op)
    #     test_acc = 0.
    #     test_count = 0
    #     for _ in range(val_batches_per_epoch):#测试集测试
    #
    #         img_batch, label_batch = sess.run(next_batch)
    #         acc = sess.run(accuracy, feed_dict={x: img_batch,
    #                                             y: label_batch,
    #                                             keep_prob: 1.})
    #         test_acc += acc
    #         test_count += 1
    #     test_acc /= test_count
    #     print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
    #                                                    test_acc))
    #     print("{} Saving checkpoint of model...".format(datetime.now()))
    #
    #     # save checkpoint of the model
    #     checkpoint_name = os.path.join(checkpoint_path,
    #                                    'model_epoch'+str(epoch+1)+'.ckpt')
    #     save_path = saver.save(sess, checkpoint_name)
    #
    #     print("{} Model checkpoint saved at {}".format(datetime.now(),
    #                                                    checkpoint_name))
