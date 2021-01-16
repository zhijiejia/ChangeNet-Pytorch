import torch
import numpy as np

colormap = [
    (255, 255, 255), (34, 177, 76), (255, 127, 39), (163, 73, 164), (255, 174, 201), (0, 162, 232), (237, 28, 36), (181, 230, 29),
    (255, 242, 0), (63, 72, 204), (136, 0, 21), (0, 0, 0)
]
weight = torch.ones(size=(1, 12)) * 3
weight[0] = 1
weight[-1] = 1
# (255, 255, 255) : background
# (0, 0, 0) : black

def decode_segmap(img, classes=0):
    '''

    :param img: N * H * W
    :param classes:
    :return: rgb-shape: N * 3 * H * W
    '''
    res = []
    img = img.cpu().numpy()
    mask_cnt, img_height, img_width = img.shape
    for cnt in range(mask_cnt):
        r = img[cnt].copy()
        g = img[cnt].copy()
        b = img[cnt].copy()
        for ll in range(0, classes):
            r[img[cnt] == ll] = colormap[ll][0]
            g[img[cnt] == ll] = colormap[ll][1]
            b[img[cnt] == ll] = colormap[ll][2]
        rgb = np.zeros((3, img_height, img_width))
        rgb[0, :, :] = r
        rgb[1, :, :] = g
        rgb[2, :, :] = b
        res.append(torch.tensor(rgb.astype(np.uint8)))
    return torch.stack(res, dim=0)


