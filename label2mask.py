import os
import numpy as np
from PIL import Image

data_root = '/root/PycharmProjects/ChangeNet/raw/'
color_list = [
    (255, 255, 255), (34, 177, 76), (255, 127, 39), (163, 73, 164), (255, 174, 201), (0, 162, 232), (237, 28, 36), (181, 230, 29),
    (255, 242, 0), (63, 72, 204), (136, 0, 21), (0, 0, 0)
]

hashc = dict()
for index, color in enumerate(color_list):
    hashc[color] = index

def new_mask(dirPath, imgName):
    img = Image.open(dirPath + imgName).convert('RGB')
    img = np.array(img)
    h, w, _ = img.shape
    mask = np.zeros(shape=(h, w))
    for i in range(h):
        for j in range(w):
            print(hashc[(img[i][j][0], img[i][j][1], img[i][j][2])])
            if (img[i][j][0], img[i][j][1], img[i][j][2]) in hashc:
                mask[i][j] = hashc[(img[i][j][0], img[i][j][1], img[i][j][2])]

    im = Image.fromarray(mask).convert('RGB')
    print(f'{dirPath}mask-{imgName}')
    im.save(f'{dirPath}mask-{imgName}')


for dir in os.listdir(data_root):
    gtPath = data_root + dir + '/GT/'
    for png in os.listdir(gtPath):
        if png.split('.')[-1] == 'png':
            new_mask(gtPath, png)
