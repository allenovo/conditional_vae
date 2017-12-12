import ipdb
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc


class VAE:

    def __init__(self, layers):
        self.__build_graph__(layers)

    def __build_graph__(self, layers):
        input_sz  = layers[0]
        latent_sz = layers[-1] / 2
        
        # encoder (parametrization of approximate posterior q(z|x))
        x = tf.placeholder(tf.float32, [None, input_sz]) # input layer
        with tf.variable_scope('encoder', reuse=False):
            fc_x = x
            for hidden in layers[1:-1]: # hidden layers
                fc_x = fc(fc_x, hidden)
            z_param = fc(fc_x, latent_sz * 2, activation_fn=None)
            z_log_sigma_sq = z_param[:,:latent_sz] # log deviation square of q(z|x)
            z_mu = z_param[:,latent_sz:]           # mean of q(z|x)

            # sample latent variable z from q(z|x)
            eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq))
            z = tf.sqrt(tf.exp(z_log_sigma_sq)) * eps + z_mu

        # decoder (parametrization of likelihood p(x|z))
        # it follows the mirror structure of encoder
        with tf.variable_scope('decoder', reuse=False):
            fc_z = z
            for hidden in layers[::-1][1:-1]: # hidden layers
                fc_z = fc(fc_z, hidden)
            x_hat = fc(fc_z, input_sz, activation_fn=tf.sigmoid) # reconstruction layer

        # loss: negative of Evidence Lower BOund (ELBO) 
        # 1. KL-divergence: KL(q(z|x)||p(z)) 
        # (divergence between two multi-variate normal distribution, please refer to wikipedia)
        kl_loss = -tf.reduce_mean(0.5 * tf.reduce_sum( \
                  1+z_log_sigma_sq-tf.square(z_mu)-tf.exp(z_log_sigma_sq), axis=1))

        # 2. Likelihood: p(x|z)
        # also called as reconstruction loss
        # we parametrized it with binary cross-entropy loss as MNIST contains binary images
        eps = 1e-10  # add small number to avoid log(0.0)
        recon_loss = tf.reduce_mean(-tf.reduce_sum( \
                      x * tf.log(eps + x_hat) + (1 - x) * tf.log(1 - x_hat + eps), axis=1))
        total_loss = kl_loss + recon_loss

        # record variables
        self.z = z
        self.total_loss, self.recon_loss, self.kl_loss = total_loss, recon_loss, kl_loss
        self.x, self.x_hat = x, x_hat

    def get_loss(self):
        return self.total_loss, self.recon_loss, self.kl_loss

    def reconstruction(self):
        return self.x_hat

    def latent_feature(self):
        return self.z



