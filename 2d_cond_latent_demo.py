from __future__ import print_function
import os
import sys
import time
import math
import ipdb
import numpy as np
import tensorflow as tf
import scipy.misc as sm
import matplotlib.pyplot as plt

from cvae import CVAE
from utilities import *
from pynput.mouse import Button, Controller
from pynput import keyboard
from pynput.keyboard import Key, Listener
from tensorflow.examples.tutorials.mnist import input_data

cond = 0
key_map = {'<29>':0, '<18>':1, '<19>':2, '<20>':3, '<21>':4, \
           '<23>':5, '<22>':6, '<26>':7, '<28>':8, '<25>':9}

def on_press(key):
    try:
        global cond
        cond = key_map['%s'%key]
        print('Showing digit %d'%cond)
    except AttributeError:
        pass


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


def start_demo():
    
    model_filename = 'cvae'
    model_save_dir = './ckpt/' + model_filename
    model_filepath = os.path.join(model_save_dir, 'cvae')

    # load MNIST dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    N = mnist.train.num_examples

    # build VAE
    layers = [28*28, 512, 256, 4] # layer configuration
    vae = CVAE(layers, 10)

    # open a training session
    sess = tf.InteractiveSession()

    # restore model
    var_list = tf.global_variables()
    # filter out weights for encoder in the checkpoint
    var_list = [var for var in var_list if ('encoder' in var.name or 'decoder' in var.name)]
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(sess, model_filepath)
    print('[*] Loading success: %s!'%model_filepath)
    
    mouse = Controller()
    x_, y_ = mouse.position[0], mouse.position[1]
    z = z_ = np.zeros((1,2))
    unit = 200

    with Listener(on_press=on_press, on_release=on_release) as listener:
        # live demo with cursor
        img = None
        while True:
            y  = np.zeros((1,10))
            y[0,cond] = 1
            im = sess.run(vae.x_hat[0], feed_dict={vae.z: z, vae.y: y})
            im = np.reshape(im, (28,28))
            if img is None:
                img = plt.imshow(im, cmap='gray')
            else:
                img.set_data(im)
            plt.pause(.01)
            plt.draw()

            x, y = mouse.position[0], mouse.position[1]
            z = z_ + [(x-x_)/unit, (y-y_)/unit]


def main():
    start_demo()


if __name__ == '__main__':
    main()