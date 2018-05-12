import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

# show our beautiful painting range
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
# plt.show()
#
#
def artist_works():     # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    return paintings


with tf.variable_scope('generator'):
    g_input = tf.placeholder(tf.float32,[None,N_IDEAS])
    g_layer1 = tf.layers.dense(g_input,128,activation=tf.nn.relu)
    g_out = tf.layers.dense(g_layer1,ART_COMPONENTS)


with tf.variable_scope('discriminator'):
    d_input = tf.placeholder(dtype=tf.float32,shape=[None,ART_COMPONENTS],name='real_in')
    d_layer1 = tf.layers.dense(d_input,128,activation=tf.nn.relu,name='d_layer1')
    probability_real = tf.layers.dense(inputs=d_layer1,units=1,activation=tf.nn.sigmoid,name='d_out_real')

    #生成器可以共享权重重新使用辨别器的权重
    #d_layer2 = tf.layers.dense(g_out,128,activation=tf.nn.relu,name='d_layer2',reuse=True)
    d_layer2 = tf.layers.dense(g_out, 128, activation=tf.nn.relu, name='d_layer1',reuse=True)
    probability_fake = tf.layers.dense(d_layer2,units=1,activation=tf.nn.sigmoid,name='d_out_real',reuse=True)

d_loss = -tf.reduce_mean(tf.log(probability_real) + tf.log(1-probability_fake))

g_loss = tf.reduce_mean(tf.log(1-probability_fake))

train_d = tf.train.AdamOptimizer(LR_D).minimize(d_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
train_g = tf.train.AdamOptimizer(LR_G).minimize(g_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   # something about continuous plotting
for step in range(5000):
    artist_paintings = artist_works() #获取样本的真实数据
    G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)#随机出假的样本数据
    G_paintings, pa0, Dl = sess.run([g_out, probability_real, d_loss, train_d, train_g],    # train and get results
                                    {g_input: G_ideas, d_input: artist_paintings})[:3]

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -Dl, fontdict={'size': 15})
        plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.draw(); plt.pause(0.01)

plt.ioff()
plt.show()