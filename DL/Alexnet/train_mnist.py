import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("../data_set/MNIST_data/",one_hot = True)
batch_size = 128
dropout_rate = 0.5


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

def train():
    x = tf.placeholder(tf.float32,[None,28,28,1],name='x')
    y_ = tf.placeholder(tf.int32,[None],name='y_')
    keep_prob = tf.placeholder(tf.float32)#dropout rate


    out_y = inference(x,keep_prob,10)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_y,labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean
    train_op = tf.train.AdamOptimizer(0.00008).minimize(loss)

    correct_prediction = tf.equal(tf.cast(tf.argmax(out_y,1),tf.int32),y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    saver = tf.train.Saver()#use saver save the model
    model_path = "model/"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_num = 2000


        x_t = mnist.test.images
        x_t = x_t.reshape(-1,28,28,1)
        y_t = np.argmax(mnist.test.labels,axis = 1)
        test_feed = {x:x_t,y_:y_t,keep_prob:1}


        x_v = mnist.validation.images
        x_v = x_v.reshape(-1,28,28,1)
        y_v = np.argmax(mnist.validation.labels,axis=1)
        validate_feed = {x:x_v,y_:y_v,keep_prob:1}

        # #restore the model data
        # ckpt = tf.train.get_checkpoint_state("path")
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, "model/model.ckpt")#reload the saved model

        for i in range(train_num):
            xs,ys = mnist.train.next_batch(batch_size)
            xs = xs.reshape(batch_size,28,28,1)
            ys = np.argmax(ys,axis = 1)

            _,losses = sess.run([train_op,loss],feed_dict={x:xs,y_:ys,keep_prob:dropout_rate})
            print("the %d step loss is %f"%(i,losses))

            if i%10 == 0:
                #saver.save(sess, "model/model.cptk")  # save the train model
                saver.save(sess,os.path.join(model_path,"model.ckpt"),global_step=10)#save the model
            #     vali_acc = sess.run(accuracy,feed_dict=validate_feed)

# train()

