import tensorflow as tf
import alexnet
import cv2
import os
import numpy as np
import classes

def evaluate():
    with tf.Graph().as_default() as g:
        # Link variable to model output
        batch_size = 1
        keep_prob = 1
        num_classes = 2
        skip = []

        mode_save_path = 'checkpoints'
        # test_file = 'test.txt'
        img_path = 'id_test_image/333/'
        images = []
        for f in os.listdir(img_path):
            images.append(cv2.imread(img_path + f))

        # 定义输入输出的格式
        imgMean = np.array([104, 117, 124], np.float)
        x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
        model = alexnet.AlexNet(x, keep_prob, num_classes, skip)
        score = model.fc8
        softmax = tf.nn.softmax(score)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(mode_save_path)
            count1 = 0
            count2 = 0
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                for i, img in enumerate(images):
                    test = cv2.resize(img.astype(np.float), (227, 227))  # resize成网络输入大小
                    test -= imgMean  # 去均值
                    test_img = test.reshape((1, 227, 227, 3))  # 拉成tensor

                    maxx = np.argmax(sess.run(softmax, feed_dict={x: test_img}))
                    res = classes.class_names[maxx]  # 取概率最大类的下标
                    print(res)
                    if res == 'id':
                        count1+=1
                    elif res == 'not':
                        count2+=1
                print("id is %d"%count1)
                print("not id is %d"%count2)
            else:
                print('No checkpoint file found')
                return

def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()
