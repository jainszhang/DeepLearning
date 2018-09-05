import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 10
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3

#draw picture
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

#weight initializer
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(dtype=tf.float32,shape=[None,X_dim])
z = tf.placeholder(dtype=tf.float32,shape=[None,z_dim])

#Encoder
Q_w1 = tf.Variable(initial_value=xavier_init([X_dim,h_dim]))
Q_b1 = tf.Variable(initial_value=tf.zeros(shape=[h_dim]))

Q_w2 = tf.Variable(initial_value=xavier_init([h_dim,z_dim]))
Q_b2 = tf.Variable(initial_value=tf.zeros(shape=[z_dim]))

theta_Q = [Q_w1,Q_w2,Q_b1,Q_b2]

def Q(x):
    Q1 = tf.nn.relu(tf.matmul(x,Q_w1) + Q_b1)
    return tf.matmul(Q1,Q_w2) + Q_b2

#Generator or Decoder
P_w1 = tf.Variable(initial_value=xavier_init([z_dim,h_dim]))
P_b1 = tf.Variable(initial_value=tf.zeros([h_dim]))

P_w2 = tf.Variable(initial_value=xavier_init([h_dim,X_dim]))
P_b2 = tf.Variable(initial_value=tf.zeros([X_dim]))

theta_P = [P_w1,P_w2,P_b1,P_b2]

def P(x):
    P1 = tf.nn.relu(tf.matmul(x,P_w1) + P_b1)
    return tf.matmul(P1,P_w2) + P_b2,tf.nn.sigmoid(tf.matmul(P1,P_w2) + P_b2)

#Discriminator
D_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def D(z):
    h = tf.nn.relu(tf.matmul(z, D_W1) + D_b1)
    logits = tf.matmul(h, D_W2) + D_b2
    prob = tf.nn.sigmoid(logits)
    return prob


z_sample = Q(X)
_,logits = P(z_sample)

X_samples,_ = P(z)

recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X))

D_real = D(z)
D_fake = D(z_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

AE_solver = tf.train.AdamOptimizer().minimize(recon_loss, var_list=theta_P + theta_Q)
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_Q)

lossd_all = []
lossg_all = []
lossre_all = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0

    for it in range(100000):
        X_mb, _ = mnist.train.next_batch(mb_size)
        z_mb = np.random.randn(mb_size, z_dim)

        _, recon_loss_curr = sess.run([AE_solver, recon_loss], feed_dict={X: X_mb})
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, z: z_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb})

        if it % 1000 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; Recon_loss: {:.4}'
                  .format(it, D_loss_curr, G_loss_curr, recon_loss_curr))

            lossd_all.append(D_loss_curr)
            lossg_all.append(G_loss_curr)
            lossre_all.append(recon_loss_curr)

            samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)


    plt.figure()
    plt.plot(lossd_all,'b',label='lossd')
    plt.plot(lossg_all,'r',label='lossg')
    plt.plot(lossre_all,'g',label='lossre_all')
    plt.legend()#展示图例

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("D and G losses")
    img_name1 = '1.jpg'
    # plt.show()
    plt.savefig(img_name1)