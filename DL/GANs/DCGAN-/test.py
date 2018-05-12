# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy import misc
import pickle



learningrate = 0.0002
epoch = 20
batch_size = 64
alpha = 0.2


def vis_img(batch_size, samples,epoch):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=8, ncols=8, sharey=True, sharex=True)

    for ax, img in zip(axes.flatten(), samples[batch_size]):
        # print(img.shape)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((96, 96, 3)), cmap='Greys_r')

    image_name = 'image/'+str(epoch)+'.jpg'
    plt.savefig(image_name)
    #plt.show()

    return fig, axes

def read_img(path):
    img = misc.imresize(misc.imread(path), size=[96, 96])
    return img

def get_batch(path, batch_size):
    img_list = [os.path.join(path, i) for i in os.listdir(path)]

    n_batchs = len(img_list) // batch_size
    img_list = img_list[:n_batchs * batch_size]

    for ii in range(n_batchs):
        tmp_img_list = img_list[ii * batch_size:(ii + 1) * batch_size]
        img_batch = np.zeros(shape=[batch_size, 96, 96, 3])
        for jj, img in enumerate(tmp_img_list):
            img_batch[jj] = read_img(img)
        yield img_batch


def generator(inputs,stddev = 0.02):
    with tf.variable_scope(name_or_scope='generator') as scope:
        fc1 = tf.layers.dense(inputs=inputs,units=64*8*6*6,name='fc1')
        reshape1 = tf.reshape(fc1,shape=(-1,6,6,512),name='reshape1')
        bn1 = tf.layers.batch_normalization(inputs=reshape1,name='bn1')
        relu1 = tf.nn.relu(bn1,name='relu1')

        deconv1 = tf.layers.conv2d_transpose(inputs=relu1,filters=256,kernel_size=[5,5],strides=2,padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=stddev),name='deconv1')
        bn2 = tf.layers.batch_normalization(inputs=deconv1,name='bn2')
        relu2 = tf.nn.relu(bn2,name='relu2')


        deconv2 = tf.layers.conv2d_transpose(inputs=relu2,filters=128,kernel_size=[5,5],strides=2,padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=stddev),name='deconv2')
        bn3 = tf.layers.batch_normalization(deconv2,name='bn3')
        relu3 = tf.nn.relu(bn3,name='relu3')


        deconv3 = tf.layers.conv2d_transpose(inputs=relu3,filters=64,kernel_size=[5,5],strides=2,padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=stddev),name='deconv3')
        bn4 = tf.layers.batch_normalization(deconv3,name='bn4')
        relu4 = tf.nn.relu(bn4,name='relu4')

        logits = tf.layers.conv2d_transpose(inputs=relu4,filters=3,kernel_size=[5,5],strides=2,padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=stddev),name='logits')
        return tf.nn.tanh(logits)


def discriminator(inputs,stddev=0.02,alpha = 0.2,batch_size = 64,name = 'discriminator',reuse = False):
    with tf.variable_scope(name_or_scope=name,reuse=reuse) as scope:
        conv1 = tf.layers.conv2d(inputs=inputs,filters=64,kernel_size=[5,5],strides=[2,2],padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=stddev),name='conv1')
        bn1 = tf.layers.batch_normalization(conv1,name='bn1')
        relu1 = tf.maximum(tf.multiply(alpha,bn1),bn1,name='relu1')


        conv2 = tf.layers.conv2d(inputs=relu1,filters=128,kernel_size=[5,5],strides=[2,2],padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=stddev),name='conv2')
        bn2 = tf.layers.batch_normalization(conv2,name='bn2')
        relu2 = tf.maximum(tf.multiply(alpha,bn2),bn2,name='relu2')



        conv3 = tf.layers.conv2d(inputs=relu2, filters=256, kernel_size=[5, 5], strides=[2, 2], padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='conv3')
        bn3 = tf.layers.batch_normalization(conv3, name='bn3')
        relu3 = tf.maximum(tf.multiply(alpha, bn3), bn3, name='relu3')


        conv4 = tf.layers.conv2d(inputs=relu3, filters=512, kernel_size=[5, 5], strides=[2, 2], padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='conv4')
        bn4 = tf.layers.batch_normalization(conv4, name='bn4')
        relu4 = tf.maximum(tf.multiply(alpha, bn4), bn4, name='relu4')

        flatten = tf.reshape(relu4,[batch_size,6*6*512],name='flatten')


        fc5 = tf.layers.dense(inputs=flatten,units=1,
                              kernel_initializer=tf.random_normal_initializer(stddev=stddev),name='fc5')

        return fc5


def train(epoch):
    g_x = tf.placeholder(shape=[None,100],dtype=tf.float32,name='g_x')

    r_x = tf.placeholder(shape=[None,96,96,3],dtype=tf.float32,name='r_x')

    g_out = generator(inputs=g_x,stddev=0.02)

    real_logits = discriminator(inputs=r_x,stddev=0.02,alpha=alpha,batch_size=batch_size,reuse=False)
    fake_logits = discriminator(inputs=g_out,reuse=True)

    train_var = tf.trainable_variables()#获取训练样本列表
    var_list_gen = [var for var in train_var if var.name.startswith('generator')]
    var_list_dis = [var for var in train_var if var.name.startswith('discriminator')]

    with tf.name_scope('metrics') as scope:
        loss_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logits) * 0.9, logits=fake_logits))

        loss_d_f = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logits), logits=fake_logits))
        loss_d_r = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logits) * 0.9, logits=real_logits))
        loss_d = loss_d_f + loss_d_r


        gen_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss_g, var_list=var_list_gen)
        dis_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss_d, var_list=var_list_dis)

        d_loss = []
        g_loss = []
        out_gloss = open('g_loss.pkl', 'wb')
        out_dloss = open('d_loss.pkl', 'wb')

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            writer = tf.summary.FileWriter('checkpoints', sess.graph)
            saver = tf.train.Saver()



            for epoch in range(epoch):
                total_g_loss = 0
                total_d_loss = 0
                KK = 0
                for batch in get_batch('../../data_set/faces/', batch_size):

                    x_real = batch
                    x_real = x_real / 127.5 - 1
                    x_fake = np.random.uniform(-1, 1, size=[batch_size, 100])

                    KK += 1

                    _, tmp_loss_d = sess.run([dis_optimizer, loss_d], feed_dict={g_x: x_fake, r_x: x_real})

                    total_d_loss += tmp_loss_d

                    _, tmp_loss_g = sess.run([gen_optimizer, loss_g], feed_dict={g_x: x_fake})
                    # _, tmp_loss_g = sess.run([gen_optimizer, loss_g], feed_dict={g_x: x_fake})
                    total_g_loss += tmp_loss_g

                    d_loss.append(total_d_loss)
                    g_loss.append(total_g_loss)
                    print("epoch is %d , each step is %d :d_loss is %f....g_loss is %f"%(epoch,KK,tmp_loss_d,tmp_loss_g))



                if epoch % 1 == 0:
                    x_fake = np.random.uniform(-1, 1, [64, 100])

                    samples = sess.run(g_out, feed_dict={g_x: x_fake})
                    samples = (((samples - samples.min()) * 255) / (samples.max() - samples.min())).astype(np.uint8)

                    vis_img(-1, [samples],epoch)

                    print('epoch {},loss_g={}'.format(epoch, total_g_loss / 2 / KK))
                    print('epoch {},loss_d={}'.format(epoch, total_d_loss / KK))


            pickle.dump(g_loss, out_gloss)
            pickle.dump(d_loss, out_dloss)
            out_gloss.close()
            out_dloss.close()

            writer.close()
            saver.save(sess, "./checkpoints/DCGAN")

train(epoch=epoch)