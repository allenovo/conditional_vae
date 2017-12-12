from __future__ import print_function
import os
import math
import ipdb
import numpy as np
import tensorflow as tf
import scipy.misc as sm

from vae import VAE
from utilities import *
from tensorflow.examples.tutorials.mnist import input_data

def run_training(num_epoch, batch_size, lr):

    model_filename = 'vae'
    model_save_dir = './ckpt/' + model_filename
    pred_save_dir  = './output/' + model_filename
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)

    # load MNIST dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    N = mnist.train.num_examples

    # build VAE
    layers = [28*28, 512, 256, 4] # layer configuration
    vae = VAE(layers)
    train_step = tf.train.AdamOptimizer(lr).minimize(vae.total_loss)

    # open a training session
    sess = tf.InteractiveSession()

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # training
    num_iter = int(math.ceil(N / batch_size))
    print("Start training ... %d iterations per epoch" %num_iter)
    for i in range(num_epoch):
        for it in range(num_iter):
            batch = mnist.train.next_batch(batch_size)
            _, total_loss, recon_loss, kl_loss = \
                sess.run([train_step, vae.total_loss, vae.recon_loss, vae.kl_loss], feed_dict={vae.x: batch[0]})
            #if it % 200 == 0:
            #    print("\tIter [%d/%d] total_loss=%.6f, recon_loss=%.6f, kl_loss=%.6f" \
            #        %(it+1, num_iter, total_loss, recon_loss, kl_loss))

        batch = mnist.train.next_batch(N)
        x_hat, total_loss, recon_loss, kl_loss = \
            sess.run([vae.x_hat[0], vae.total_loss, vae.recon_loss, vae.kl_loss], feed_dict={vae.x: batch[0]})
        print("Epoch [%d/%d] total_loss=%.6f, recon_loss=%.6f, kl_loss=%.6f" \
              %(i+1, num_epoch, total_loss, recon_loss, kl_loss))
        # save reconstructed image
        x = np.reshape(batch[0][0], (28,28))
        x_hat = np.reshape(x_hat, (28,28))
        sm.imsave(os.path.join(pred_save_dir, '%07d.jpg'%i), montage([x,x_hat], [1,2]))

        if (i+1) % 50 == 0:
            # save model
            saver = tf.train.Saver(max_to_keep=10)
            saver.save(sess, os.path.join(model_save_dir, model_filename), global_step=i+1)
            print("Model saved.")


def main():
    num_epoch = 1000
    batch_size = 128
    lr = 1e-3
    run_training(num_epoch, batch_size, lr)


if __name__ == '__main__':
    main()