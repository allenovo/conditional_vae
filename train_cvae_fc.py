from __future__ import print_function
import os
import math
import ipdb
import numpy as np
import tensorflow as tf
import scipy.misc as sm
import matplotlib.pyplot as plt

from cvae import CVAE
from utilities import *

def run_training(num_epoch, batch_size, lr):

    model_filename = 'cvae_face'
    model_save_dir = './ckpt/' + model_filename
    pred_save_dir  = './output/' + model_filename
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)

    # load yale face dataset
    face_size = 64
    X, Y = load_faces('yale_face', ['yaleB01','yaleB03','yaleB05','yaleB07','yaleB09'], size=face_size)
    X_norm, Y_norm = X/255., Y/255.
    N = X.shape[0]
    X_norm = np.reshape(X_norm, [N, -1])
    Y_norm = np.reshape(Y_norm, [N, -1])
    ipdb.set_trace()

    # build VAE
    layers = [face_size*face_size, 512, 256, 4] # layer configuration
    vae = CVAE(layers, face_size*face_size)
    train_step = tf.train.AdamOptimizer(lr).minimize(vae.total_loss)

    # open a training session
    sess = tf.InteractiveSession()

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # training
    num_iter = int(math.ceil(N / batch_size))
    print("Start training ... %d iterations per epoch" %num_iter)
    for i in range(num_epoch):
        rand_idx = np.random.permutation(N)
        for it in range(num_iter):
            idx = rand_idx[it*batch_size:(it+1)*batch_size]
            X_batch, Y_batch = X_norm[idx,::], Y_norm[idx,::]
            _, total_loss, recon_loss, kl_loss = \
                sess.run([train_step, vae.total_loss, vae.recon_loss, vae.kl_loss], feed_dict={vae.x: X_batch, vae.y:Y_batch})
            #if it % 1 == 0:
            #    print("\tIter [%d/%d] total_loss=%.6f, recon_loss=%.6f, kl_loss=%.6f" \
            #        %(it+1, num_iter, total_loss, recon_loss, kl_loss))

        x_hat, total_loss, recon_loss, kl_loss = \
            sess.run([vae.x_hat[rand_idx[0]], vae.total_loss, vae.recon_loss, vae.kl_loss], feed_dict={vae.x: X_norm, vae.y:Y_norm})
        print("Epoch [%d/%d] total_loss=%.6f, recon_loss=%.6f, kl_loss=%.6f" \
              %(i+1, num_epoch, total_loss, recon_loss, kl_loss))
        # save reconstructed image
        x = X[rand_idx[0],:,:,0]
        x_hat = np.reshape(x_hat, (face_size,face_size))*255
        sm.imsave(os.path.join(pred_save_dir, '%07d.jpg'%(i+1)), montage([x,x_hat], [1,2]))

        # save model
        if (i+1) % 200 == 0:
            saver = tf.train.Saver(max_to_keep=10)
            saver.save(sess, os.path.join(model_save_dir, model_filename))


def main():
    num_epoch = 1000
    batch_size = 16
    lr = 1e-3
    run_training(num_epoch, batch_size, lr)


if __name__ == '__main__':
    main()