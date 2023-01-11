import random

import os
import torchvision.transforms as T
from torchvision.io import read_image
import torch
import glob
import numpy as np
import pickle

IMG_DIR = '/data/pwojcik/images_he_seg/positive/'
IMG_SIZE = 512


def get_bounding_boxes(boxes, x_offset, y_offset):
    idx = []
    for i in range(boxes.shape[0]):
        if (boxes[i, 0] >= y_offset) and (boxes[i, 1] >= x_offset) \
                and (boxes[i, 2] < y_offset + IMG_SIZE) and (boxes[i, 3] < x_offset + IMG_SIZE):
            idx.append(i)
    return idx


if __name__ == '__main__':
    files = glob.glob(os.path.join(IMG_DIR, '*'))
    files = [f for f in files if not 'bis' in f and not 'aug' in f and not 'pkl' in f]
    c, r = [], []
    for f in files:
        n = f.split('-')
        c.append(int(n[-1].strip('.png')[1:]))
        r.append(int(n[-2][1:]))
    mc = max(c)
    mr = max(r)
    for ic in range(mc + 1):
        for jr in range(mr + 1):
            lu = ic, jr
            ru = ic + 1, jr
            ld = ic, jr + 1
            rd = ic + 1, jr + 1
            flu = f"wsi_001-tile-r{lu[0]}-c{lu[1]}.png"
            flu_p = f"wsi_001-tile-r{lu[0]}-c{lu[1]}.pkl"
            fru = f"wsi_001-tile-r{ru[0]}-c{ru[1]}.png"
            fru_p = f"wsi_001-tile-r{ru[0]}-c{ru[1]}.pkl"
            fld = f"wsi_001-tile-r{ld[0]}-c{ld[1]}.png"
            fld_p = f"wsi_001-tile-r{ld[0]}-c{ld[1]}.pkl"
            frd = f"wsi_001-tile-r{rd[0]}-c{rd[1]}.png"
            frd_p = f"wsi_001-tile-r{rd[0]}-c{rd[1]}.pkl"
            files = [flu, fru, fld, frd]
            pickle_files = [flu_p, fru_p, fld_p, frd_p]
            all = True
            for f in files:
                if not os.path.exists(os.path.join(IMG_DIR, f)):
                    all = False
                    break
            if not all:
                continue
            pickles = [np.load(os.path.join(IMG_DIR, f), allow_pickle=True) for f in pickle_files]
            pickles[1][0::2] += IMG_SIZE
            pickles[2][1::2] += IMG_SIZE

            pickles[3][0::2] += IMG_SIZE
            pickles[3][1::2] += IMG_SIZE

            images = [read_image(os.path.join(IMG_DIR, f)) for f in files]
            image1 = torch.cat([images[0], images[1]], dim=1)
            image2 = torch.cat([images[2], images[3]], dim=1)
            image = torch.cat([image1, image2], dim=2)
            #image_boxes = T.ToPILImage()(image)
            #canvas = ImageDraw.Draw(image_boxes)
            offsets = [(random.randint(10, IMG_SIZE), random.randint(10, IMG_SIZE)) for _ in range(10)]
            for cc in range(10):
                a = random.randint(10, IMG_SIZE)
                b = random.randint(10, IMG_SIZE)
                idx = get_bounding_boxes(torch.cat(pickles), a, b)
                pickles_n = [np.load(os.path.join(IMG_DIR, f), allow_pickle=True) for f in pickle_files]
                pickles_n = torch.cat(pickles_n)
                pckls = [pickles_n[i] for i in idx]

                crop = image[:, a:(a + IMG_SIZE), b:(b + IMG_SIZE)]
                shape = [a, b, a + IMG_SIZE, b + IMG_SIZE]
                #canvas.rectangle(shape, outline="blue")
                assert(crop.shape == (3, IMG_SIZE, IMG_SIZE))
                crop = T.ToPILImage()(crop)
                #crop.save(os.path.join(IMG_DIR, f"wsi_001-tile-r{a}-c{b}aug.png"))
                with open(os.path.join(IMG_DIR, f"wsi_001-tile-r{a}-c{b}aug.pkl"), 'wb') as outf:
                    print(torch.cat(pckls).shape)
                    #pickle.dump(torch.cat(pckls), outf)
                #crop.show()
            #image_boxes.show()



