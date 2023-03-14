from typing import List

import numpy as np
import os
from os import listdir
from os.path import isfile, join
import scipy
import torch
import random


def create_bounding_boxes():
    bboxes = []
    for vid, value in enumerate(obj_ids.numpy().tolist()):
        mask = inst_map == value
        if not torch.any(mask):
            continue
        insta_map_v = inst_map * mask.int()
        lbl = label(insta_map_v)
        props = regionprops(lbl)
        for prop in props:
            bbox = prop.bbox

            b = torch.tensor([bbox[1], bbox[0], bbox[3], bbox[2]]).unsqueeze(0)
            bboxes.append((b, idx_c[vid]))

    random.shuffle(bboxes)
    bboxes, classes = zip(*bboxes)
    boxes = torch.cat(bboxes, dim=0)
    return boxes, list(classes)


def transform(input_dir, output_dir):
    '''
    Input folder should contain following files
        XXX.png -> dataset images,
        XXX.mat -> Hovernet predictions,
        XXX.pkl -> indexes of paired nuclei from Hovernet,
        XXX_cls.pkl -> classes of paired nuclei
    '''

    def _get_files(path, predicate) -> List[str]:
        return [f for f in listdir(path) if isfile(join(path, f)) and predicate(f)]

    masks_files = _get_files(input_dir, lambda file_path: file_path.endswith('.mat'))
    masks_files.sort()

    for mask_file in masks_files:
        mat = scipy.io.loadmat(os.path.join(input_dir, mask_file))
        inst_map = torch.tensor(mat['inst_map'])
        idx, idx_c = [np.load(os.path.join(input_dir, mask_file.replace('.mat', suffix)), allow_pickle=True) for suffix
                      in ('.pkl', '_cls.pkl')]

        # we avoid background
        obj_ids = torch.unique(inst_map)[1:]

        boxes, classes = create_bounding_boxes(inst_map, obj_ids[idx], idx_c)

        file = mask_file.strip('.mat')

    pass