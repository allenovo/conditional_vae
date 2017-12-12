import os
import math
import ipdb
import numpy as np
import scipy.misc as sm

from os import listdir


def montage(imgs, shape):
    assert len(imgs) == np.prod(shape)
    w, h = imgs[0].shape[0], imgs[0].shape[1]
    montage_img = np.zeros((h*shape[0], w*shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            img = imgs[i*shape[1]+j]
            montage_img[i*h:(i+1)*h,j*w:(j+1)*w] = img

    return montage_img


def normalize(imgs):
    return imgs / 255. * 2 - 1


def denormalize(imgs):
    return (imgs + 1) * 255 / 2.


def load_image(path, size=64):
    im = sm.imread(path)
    im = sm.imresize(im, (size,size))
    return np.reshape(im, (1, size, size, 1))


def load_faces(base, face_names, size=64):
    X, Y = [], []
    for face_name in face_names:
        path = os.path.join(base, face_name)
        face_imgs = [f for f in listdir(path) if f.endswith('.pgm') and len(f.split('_')) == 2]
        Y_face_img = face_name + '_P00A+000E+00.pgm'
        if Y_face_img not in face_imgs:
            print("No condition face for %s!"%face_name)
            return X, Y

        Y_face = load_image(os.path.join(path, Y_face_img), size)
        for face_img in face_imgs:
            X.append(load_image(os.path.join(path, face_img), size))
            Y.append(Y_face)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    return X, Y


