from typing import List

import torch
from os.path import isfile, join
from os import listdir
import os
from skimage.measure import label, regionprops
import scipy
import pickle
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import random

IMG_PATH='/Users/piotrwojcik/Downloads/Fold 1/images/fold1/images.npy'
MAKS_PATH='/Users/piotrwojcik/Downloads/Fold 1/masks/fold1/masks.npy'
IMG_OUTPUT_DIR = '/Users/piotrwojcik/Downloads/pannuke/img_dir/train/'
ANN_OUTPUT_DIR = '/Users/piotrwojcik/Downloads/pannuke/ann_dir/train/'


ratio = 448 / 256

from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json


def create_bboxes_for_mask(mask):
    bboxes = []
    for cls in range(5):
        m = mask[cls]
        obj_ids = torch.unique(m)
        for value in obj_ids.numpy().tolist():
            mm = m == value
            insta_map_v = m * mm.int()
            lbl = label(insta_map_v)
            props = regionprops(lbl)

            for prop in props:
                bbox = prop.bbox
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < 5:
                    continue

                b = torch.tensor([bbox[1], bbox[0], bbox[3], bbox[2]]).unsqueeze(0)
                bboxes.append((b, cls))
    if not len(bboxes):
        print('WARN: empty')
        return None, None
    random.shuffle(bboxes)
    bboxes, classes = zip(*bboxes)
    boxes = torch.cat(bboxes, dim=0)
    boxes = boxes.float()
    boxes[:, 0::2] *= ratio
    boxes[:, 1::2] *= ratio
    classes = torch.tensor(classes)
    assert(boxes.shape[0] == classes.shape[0])
    return boxes.int(), classes


types = ['neoplastic ', 'inflammatory', 'soft', 'dead', 'epithelial']


def generate_coco_file(masks, images, coco):
    for m in range(10):
        mask = masks[m]
        img = images[m]
        img = img.float()
        img = img / 255
        img = F.resize(img, size = (448, 448))
        boxes, classes = create_bboxes_for_mask(mask)
        boxes = boxes.float()
        if boxes is None:
            continue
        img = T.ToPILImage()(img)
        img.save(os.path.join(IMG_OUTPUT_DIR, f"{m}.png"))

        coco_image = CocoImage(file_name=os.path.join(IMG_OUTPUT_DIR, f"{m}.png"), height=448, width=448)
        for b in range(boxes.shape[0]):
            x_min = boxes[b, 0].item()
            y_min = boxes[b, 1].item()
            x_max = boxes[b, 2].item()
            y_max = boxes[b, 3].item()
            category_id = classes[b].item()

            coco_image.add_annotation(CocoAnnotation(
                    bbox=[boxes[b, 0], boxes[b, 1], x_max - x_min, y_max - y_min],
                    category_id=category_id,
                    category_name=types[category_id]))
        coco.add_image(coco_image)


if __name__ == '__main__':
    images = torch.tensor(np.load(IMG_PATH, allow_pickle=True))
    images = images.permute(0, 3, 1, 2)
    images = images.to(torch.uint8)
    masks = torch.tensor(np.load(MAKS_PATH, allow_pickle=True))
    masks = masks.permute(0, 3, 1, 2)
    print('Start')
    coco = Coco()
    for id, cat in enumerate(types):
        coco.add_category(CocoCategory(id=id, name=types[id]))
    generate_coco_file(masks, images, coco)
    save_json(data=coco.json, save_path=os.path.join(ANN_OUTPUT_DIR, f"{m}.png"))