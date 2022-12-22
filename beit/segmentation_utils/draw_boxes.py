import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == '__main__':
    img_path = '/Users/piotrwojcik/Downloads/wsi_002-tile-r99bis-c89'
    img = read_image(img_path + '.png')

    bboxes = np.load(img_path + '.pkl', allow_pickle=True)
    bboxes_orig = np.load(img_path + '_orig.pkl', allow_pickle=True)

    drawn_boxes = draw_bounding_boxes(img, bboxes, colors="red")
    drawn_boxes_orig = draw_bounding_boxes(img, bboxes_orig, colors="blue")
    show(drawn_boxes)
    show(drawn_boxes_orig)
    plt.show()



