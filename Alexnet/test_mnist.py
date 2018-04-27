# import tensorflow as tf
# import alenet
# import cv2
# import os
# import numpy as np
# import classes
#
# def evaluate():
#     with tf.Graph().as_default() as g:
#         # Link variable to model output
#         batch_size = 1
#         keep_prob = 1
#         num_classes = 2
#         skip = []
#
#         mode_save_path = 'models'
#         # test_file = 'test.txt'
#         img_path = 'img_test/'
#         images = []
#         for f in os.listdir(img_path):
#             images.append(cv2.imread(img_path + f))
#
#         # 定义输入输出的格式
#         imgMean = np.array([104, 117, 124], np.float)
#         x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
#
#         model = alenet.AlexNet(x, keep_prob, num_classes, skip)
#         score = model.fc8
#         softmax = tf.nn.softmax(score)
#
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             saver = tf.train.Saver()
#             ckpt = tf.train.get_checkpoint_state(mode_save_path)
#
#             if ckpt and ckpt.model_checkpoint_path:
#                 # 加载模型
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 for i, img in enumerate(images):
#                     test = cv2.resize(img.astype(np.float), (227, 227))  # resize成网络输入大小
#                     test -= imgMean  # 去均值
#                     test_img = test.reshape((1, 227, 227, 3))  # 拉成tensor
#
#                     maxx = np.argmax(sess.run(softmax, feed_dict={x: test_img}))
#                     res = classes.class_names[maxx]  # 取概率最大类的下标
#                     print(res)
#             else:
#                 print('No checkpoint file found')
#                 return
#
# def main(argv=None):
#     evaluate()
#
#
# if __name__ == '__main__':
#     tf.app.run()

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("../data_set/MNIST_data/",one_hot = True)
batch = 500


def inference(X, KEEP_PROB, NUM_CLASSES):
    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    conv1 = conv(X, [1, 1, 1, 96], [1, 1, 1, 1], padding='VALID', name='conv1')
    norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
    pool1 = max_pool(norm1, 1, 1, 1, 1, padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    conv2 = conv(pool1, [5, 5, 96, 256], [1, 1, 1, 1], name='conv2')
    norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
    pool2 = max_pool(norm2, 4, 4, 2, 2, padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(pool2, [3, 3, 256, 384], [1, 1, 1, 1], name='conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, [3, 3, 384, 384], [1, 1, 1, 1], name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, [3, 3, 384, 256], [1, 1, 1, 1], name='conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
    fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
    dropout6 = dropout(fc6, KEEP_PROB)

    # 7th Layer: FC (w ReLu) -> Dropout
    fc7 = fc(dropout6, 4096, 4096, name='fc7')
    dropout7 = dropout(fc7, KEEP_PROB)

    # 8th Layer: FC and return unscaled activations
    fc8 = fc(dropout7, 4096, NUM_CLASSES, relu=False, name='fc8')

    return fc8


def conv(input_tensor,filter,strides,name,padding = "SAME"):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights',filter,initializer=tf.truncated_normal_initializer(stddev=0.08))
        biases = tf.get_variable('biases',filter[3],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,weights,strides=strides,padding=padding)
        return tf.nn.relu(tf.nn.bias_add(conv1,biases))

def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        weights = tf.get_variable('weights', shape=[num_in, num_out],initializer=tf.truncated_normal_initializer(stddev=0.08),
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out],initializer=tf.truncated_normal_initializer(stddev=0.08), trainable=True)

        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)

def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)





mode_save_path = 'model'
x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
y_ = tf.placeholder(tf.int32, [None], name='y_')
# keep_prob = tf.placeholder(tf.float32)  # dropout rate

out_y = inference(x, 1, 10)

correct_prediction = tf.equal(tf.cast(tf.argmax(out_y,1),tf.int32),y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# softmax = tf.nn.softmax(out_y)

x_t = mnist.test.images
x_t = x_t.reshape(-1, 28, 28, 1)
y_t = np.argmax(mnist.test.labels, axis=1)

#test_feed = {x: x_t, y_: y_t, keep_prob: 1}
loops  = int(x_t.shape[0] / batch)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(mode_save_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        sumacc = 0.0
        for i in range(loops):
            x_batch = x_t[batch*i:batch*(i+1)]
            y_batch = y_t[batch*i:batch*(i+1)]
            output = sess.run(accuracy,feed_dict={x:x_batch,y_:y_batch})
            print("now the %d acc is %f "%(i,output))
            sumacc += output
            print("sum acc is %f"%sumacc)
        print("all of the acc is %f"%(sumacc/loops))
    #test_acc = sess.run(accuracy, feed_dict=test_feed)
    #










