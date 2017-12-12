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

from vae import VAE
from utilities import *
from pynput.mouse import Button, Controller
from pynput import keyboard
from tensorflow.examples.tutorials.mnist import input_data


def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))


def on_release(key):
    print('{0} released'.format(key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False


def start_demo():
    
    model_filename = 'vae'
    model_save_dir = './ckpt/' + model_filename
    model_filepath = os.path.join(model_save_dir, 'vae-100')

    # load MNIST dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    N = mnist.train.num_examples

    # build VAE
    layers = [28*28, 512, 256, 128, 4] # layer configuration
    vae = VAE(layers)

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
    unit = 220

    # live demo with cursor
    img = None
    try:
        while True:
            im = sess.run(vae.x_hat[0], feed_dict={vae.z: z})
            im = np.reshape(im, (28,28))
            if img is None:
                img = plt.imshow(im)
            else:
                img.set_data(im)
            plt.pause(.01)
            plt.draw()

            x, y = mouse.position[0], mouse.position[1]
            z = z_ + [(x-x_)/unit, (y-y_)/unit]
    except KeyboardInterrupt:
        pass

    # visualize 2d latent space
    n = 20
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)

    I_latent = np.empty((28*n, 28*n))
    for i, yi in enumerate(x):
        for j, xi in enumerate(y):
            z = np.array([[xi, yi]])
            x_hat = sess.run(vae.x_hat, feed_dict={vae.z:z})
            I_latent[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x_hat[0].reshape(28, 28)

    sm.imsave('latent2d.jpg', I_latent)
        

def main():
    start_demo()


if __name__ == '__main__':
    main()