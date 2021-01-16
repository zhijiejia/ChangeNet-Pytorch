import os
import os.path
import cv2
import numpy as np
from torch.utils.data import Dataset


def make_dataset(split='train', data_list=None):
    '''
        :param data_root:
        :param data_list: the txt file path
        :return: a list contains each pair of (image_path and label_path) in dataset
    '''
    image_label_list = []
    list_read = open(data_list).readlines()
    total_pairs = 0
    for line in list_read:
        line = line.strip()
        pairs = len(os.listdir(line + '/RGB')) // 2
        total_pairs += pairs
        for index in range(pairs):
            refer_path = line + '/RGB/1_{0:02d}.png'.format(index)
            test_path = line + '/RGB/2_{0:02d}.png'.format(index)
            label_path = line + '/GT/mask-gt{0:02d}.png'.format(index)
            item = (refer_path, test_path, label_path)
            image_label_list.append(item)
    print(f"Total checking {total_pairs} pair for {split} set!")
    return image_label_list


class SemDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.split = split
        if split == 'train':
            data_list = 'train.txt'
        else:
            data_list = 'test.txt'
        self.data_list = make_dataset(split, data_list=data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def read(self, path):
        image = cv2.imread(path)                              # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        # convert cv2 read image from BGR order to RGB order
        return image

    def __getitem__(self, index):
        refer_path, test_path, label_path = self.data_list[index]
        refer_image = self.read(refer_path)                   # BGR 3 channel ndarray wiht shape H * W * 3
        test_image = self.read(test_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        return self.transform(refer_image, test_image, label)


if __name__ == '__main__':
    trainDataSet = SemDataset(split='test', transform=None)