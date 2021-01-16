import random
import math
import numpy as np
import numbers
import collections
import cv2
import torch


class Compose(object):
    """
        Example:
            transform.Compose(
                [
                 transform.RandScale([0.5, 2.0]),
                 transform.ToTensor()
                 ]
            )
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, refer_image, test_image, label):
        for t in self.transform:
            refer_image, test_image, label = t(refer_image, test_image, label)
        return refer_image, test_image, label


class ToTensor(object):
    """
        1. Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
        2. Not div 255
        3. output: image-type:float, label-type: long
    """
    def __call__(self, refer_image, test_image, label):

        refer_image = torch.from_numpy(refer_image.transpose((2, 0, 1)))    # image shape : from H * W * C --> C * H * W
        # if not isinstance(refer_image, torch.FloatTensor):
        #     refer_image = refer_image.float()

        test_image = torch.from_numpy(test_image.transpose((2, 0, 1)))    # image shape : from H * W * C --> C * H * W
        # if not isinstance(test_image, torch.FloatTensor):
        #     test_image = test_image.float()

        label = torch.from_numpy(label)                         # label shape : H * W
        if not isinstance(label, torch.LongTensor):             # 这样安排label和image的shape是为了交叉熵方便
            label = label.long()
        return refer_image, test_image, label


class Normalize(object):
    """
        Normalize tensor with mean and standard deviation along channel:
            channel = (channel - mean) / std
    """
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, refer_image, test_image, label):
        if self.std is None:
            for t, m in zip(refer_image, self.mean):
                t = t.float()
                t.sub_(m)
            for t, m in zip(test_image, self.mean):
                t = t.float()
                t.sub_(m)
        else:
            for t, m, s in zip(refer_image, self.mean, self.std):
                t = t.float()
                t.sub_(m).div_(s)
            for t, m, s in zip(test_image, self.mean, self.std):
                t = t.float()
                t.sub_(m).div_(s)

        return refer_image, test_image, label


class Resize(object):
    """
        Resize the input to the given size, 'size' is a 2-element tuple in the order of (h, w)
    """

    def __init__(self, size: tuple):
        assert len(size) == 2
        self.size = size

    def __call__(self, refer_image, test_image, label):
        refer_image = cv2.resize(refer_image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        test_image = cv2.resize(test_image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return refer_image, test_image, label

class RandomHorizontalFlip(object):
    """
        function: HorizontalFlip
        p: Random probability, should be a float number in [0, 1]
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, refer_image, test_image, label):
        if random.random() < self.p:
            refer_image = cv2.flip(refer_image, 1)
            test_image = cv2.flip(test_image, 1)
            label = cv2.flip(label, 1)
        return refer_image, test_image, label


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, refer_image, test_image, label):
        if random.random() < self.p:
            refer_image = cv2.flip(refer_image, 0)
            test_image = cv2.flip(test_image, 0)
            label = cv2.flip(label, 0)
        return refer_image, test_image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, refer_image, test_image, label):
        if random.random() < 0.5:
            refer_image = cv2.GaussianBlur(refer_image, (self.radius, self.radius), 0)
            test_image = cv2.GaussianBlur(test_image, (self.radius, self.radius), 0)
        return refer_image, test_image, label