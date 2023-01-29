import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.io import read_image

ASSETS_DIRECTORY = "assets"

plt.rcParams["savefig.bbox"] = "tight"


img_path = '/Users/piotrwojcik/Downloads/attn_dump/attn_305.png'
attn_path = '/Users/piotrwojcik/Downloads/attn_dump/attn_305.pickle'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

heads=[]
heads_num = 12

if __name__ == '__main__':
    img = read_image(img_path)
    attn = np.load(attn_path, allow_pickle=True)
    img[:, ::16, :] = 0
    img[:, :, ::16] = 0

    p = (9, 24)

    cls = False

    if not cls:
        for i in range(16):
            for j in range(16):
                img[2, 16 * p[0] + i, 16 * p[1] + j] = 0
                img[0::2, 16 * p[0] + i, 16 * p[1] + j] = 0

    attn = np.load(attn_path, allow_pickle=True).squeeze()

    attn_vis = [img]

    for i in range(2):
        if cls:
            attn_i = attn[i, 0, :-100]
        else:
            attn_i = attn[i, p[0] * 28 + p[1] + 1, :-100]
        attn_i = attn_i[1:]
        attn_i = attn_i.view(28, 28)

        attn_i *= 5000
        attn_i /= torch.max(attn_i)
        attn_i *= 255

        attn_i = attn_i.int()[None, :]

        img_i = img.clone()
        #for i in range(448):
        #    for j in range(448):
        #        bi = i // 16
        #        bj = j // 16
        #        img_i[:, i, j] =
        attn_vis.append(attn_i)
    show(attn_vis)
    plt.show()


