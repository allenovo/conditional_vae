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
key_map = {'<29>':0, '<18>':1, '<19>':2, '<20>':3, '<21>':4}
people = ['yaleB01','yaleB03','yaleB05','yaleB07','yaleB09']

def on_press(key):
    try:
        global cond
        cond = key_map['%s'%key]
        print('Showing %s in dataset'%people[cond])
    except AttributeError:
        pass


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


def start_demo():
    
    model_filename = 'cvae_face'
    model_save_dir = './ckpt/' + model_filename
    model_filepath = os.path.join(model_save_dir, 'cvae_face')

    # load yale face dataset
    face_size = 64
    X, Y = load_faces('yale_face', people, size=face_size)
    X_norm, Y_norm = X/255., Y/255.
    N = X.shape[0]
    X_norm = np.reshape(X_norm, [N, -1])
    Y_norm = np.reshape(Y_norm, [N, -1])

    # build VAE
    layers = [face_size*face_size, 512, 256, 4] # layer configuration
    vae = CVAE(layers, face_size*face_size)

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
            y  = Y_norm[[64*cond],::]
            im = sess.run(vae.x_hat[0], feed_dict={vae.z: z, vae.y: y})
            im = np.reshape(im, (face_size,face_size))
            if img is None:
                img = plt.imshow(im, cmap='gray')
            else:
                img.set_data(im)
            plt.pause(.01)
            plt.draw()

            x, y = mouse.position[0], mouse.position[1]
            z = z_ - [(x-x_)/unit, (y-y_)/unit]
    


def main():
    start_demo()


if __name__ == '__main__':
    main()