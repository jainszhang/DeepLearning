import argparse
import custom_vgg19 as vgg19
import logging
import numpy as np
import os
import tensorflow as tf
import time
import utils
from functools import reduce
import numpy as np



#different layers extract different features
CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

EPOCHS = 300
LEARNING_RATE = .02
TOTAL_VARIATION_SMOOTHING = 1.5
NORM_TERM = 6.


#losses term weights(prob)
CONTENT_WEIGHT = 1.
STYLE_WEIGHT = 3.
NORM_WEIGHT = .1
TV_WEIGHT = .1

#image path
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
OUT_PATH = DIR_PATH + '../output/out_%.0f.jpg' % time.time()
INPUT_PATH, STYLE_PATH = None, None


PRINT_TRAINING_STATUS = True
PRINT_N = 100

# Logging config
log_dir = DIR_PATH + '/../log/'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
    print('Directory "%s" was created for logging.' % log_dir)
log_path = ''.join([log_dir, str(time.time()), '.log'])
logging.basicConfig(filename=log_path, level=logging.INFO)
print("Printing log to %s" % log_path)







def parse_args():
    global INPUT_PATH, STYLE_PATH, OUT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to the input image you'd like to apply the style to")
    parser.add_argument("style", help="path to the image you'd like to reference the style from")
    parser.add_argument("--out", default=OUT_PATH, help="path to where the styled image will be created")
    args = parser.parse_args()

    # Assign image paths from the arg parsing
    INPUT_PATH = os.path.realpath(args.input)
    STYLE_PATH = os.path.realpath(args.style)
    OUT_PATH = os.path.realpath(args.out)

with tf.Session() as sess:
    parse_args()#about path of images and parse paras

    photo, image_shape = utils.load_image(INPUT_PATH)#load image
    image_shape = [1] + image_shape
    photo = photo.reshape(image_shape).astype(np.float32)

    art = utils.load_image2(STYLE_PATH, height=image_shape[1], width=image_shape[2])
    art = art.reshape(image_shape).astype(np.float32)

    # Initialize the variable image that will become our final output as random noise
    noise = tf.Variable(tf.truncated_normal(image_shape, mean=.5, stddev=.1))

    # VGG Networks Init
    with tf.name_scope('vgg_content'):
        content_model = vgg19.Vgg19()
        content_model.build(photo, image_shape[1:])



tf.placeholder(shape=None,dtype=np.float)



