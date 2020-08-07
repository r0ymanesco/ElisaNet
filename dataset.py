import glob
import cv2
import ipdb
import random
import numpy as np
from util import np_to_torch

import torch
import torch.utils.data as data

import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian


class ElisaDataset(data.Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
        self.get_img_list()

    def get_img_list(self):
        self.img_list = []
        for fn in glob.iglob(self.data_dir + '/*.jpg'):
            self.img_list.append(fn)

        num_imgs = len(self.img_list)
        train_size = int(num_imgs // 1.25)
        eval_size = int(num_imgs // 10)
        random.Random(4).shuffle(self.img_list)
        if self.mode == 'TRAIN':
            self.img_list = self.img_list[:train_size]
        elif self.mode == 'VALIDATE':
            self.img_list = self.img_list[train_size:train_size+eval_size]
        elif self.mode == 'EVALUATE':
            self.img_list = self.img_list[train_size+eval_size:train_size+2*eval_size]
        else:
            raise NotImplementedError('Mode not valid')
        print('Number of images loaded: {}'.format(len(self.img_list)))

    def rotate(self, img):
        angle = np.random.randint(0, 360)
        rotated = rotate(img, angle=angle, mode='edge')
        return rotated

    def shift(self, img):
        dx = np.random.randint(-25, 26)
        dy = np.random.randint(-25, 26)
        transform = AffineTransform(translation=(dx, dy))
        shifted = warp(img, transform, mode='edge')
        return shifted

    def flip(self, img):
        type = np.random.randint(0, 2)
        if type == 0:
            flipped = np.fliplr(img)
        else:
            flipped = np.flipud(img)
        return flipped

    def add_noise(self, img):
        var = np.random.rand(1) * 0.01
        noisy = random_noise(img, var=var)
        return noisy

    # def blurr(self, img):
    #     var = np.random.rand(1)
    #     blurred = gaussian(img, sigma=var, multichannel=True)
    #     return blurred

    def __getitem__(self, index):
        fn = self.img_list[index]
        img = io.imread(fn)
        img = img[456:-456, :, :]
        label = fn.split('_')[1]
        label = float(label)
        assert label in [0., 1.]
        label = torch.tensor([label])
        mode = np.random.randint(0, 5)
        if mode == 1:
            img = self.rotate(img)
        elif mode == 2:
            img = self.shift(img)
        elif mode == 3:
            img = self.flip(img)
        elif mode == 4:
            img = self.add_noise(img)
        # elif mode == 5:
        #     img = self.blurr(img)

        img = np_to_torch(img) / 255.0
        return img, label

    def __len__(self):
        return len(self.img_list)
