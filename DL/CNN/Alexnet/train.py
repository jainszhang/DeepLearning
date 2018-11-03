import os

import numpy as np
import tensorflow as tf
import random
from 经典网络.Alexnet import Alexnet
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
num_epochs = 10
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 10


train_layers = ['fc8', 'fc7', 'fc6']

display_step = 20

filewriter_path = "tensorboard"
checkpoint_path = "model/"

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)





#申请位置
x = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)#dropout rate

model = Alexnet.AlexNet(x, keep_prob, num_classes, train_layers)
score = model.fc8

# #列出想要训练的层数--此处训练最后3层全连接层
# var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]


with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)


model_path = "model/"

#
# with tf.name_scope("train"):
#     # Get gradients of all trainable variables
#     gradients = tf.gradients(loss, var_list)
#     gradients = list(zip(gradients, var_list))
#
#     # Create optimizer and apply gradient descent to the trainable variables
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     train_op = optimizer.apply_gradients(grads_and_vars=gradients)
#
# for gradient, var in gradients:
#     tf.summary.histogram(var.name + '/gradient', gradient)
#
# for var in var_list:
#     tf.summary.histogram(var.name, var)
#
# # Add the loss to summary
# tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()
#
# # Get the number of training/validation steps per epoch
# train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
# val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))





X_train, Y_train, X_test, Y_test  = read_data.load_CIFAR10(cifar10_dir)

def shulff(X_train,Y_train):
    number = len(X_train)
    print(random.randint(0,number),number)


    return 0,0


x_train,y_train = shulff(X_train,Y_train)

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())#初始化所有变量

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # print("{} Start training...".format(datetime.now()))
    # print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
    #                                                   filewriter_path))

    # Loop over number of epochs
    # for epoch in range(num_epochs):#多少个epoch，迭代多少次

    #
    #     print("{} Epoch number: {}".format(datetime.now(), epoch+1))
    #
    #     # Initialize iterator with the training dataset
    #     sess.run(train_op)
    #
    #     for step in range(train_batches_per_epoch):
    #
    #         # get next batch of data
    #         img_batch, label_batch = sess.run(next_batch)
    #
    #         # And run the training op
    #         sess.run(train_op, feed_dict={x: img_batch,
    #                                       y: label_batch,
    #                                       keep_prob: dropout_rate})
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
