from pathlib import Path

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

IMG_PATH='/Users/piotrwojcik/Downloads/Fold 3/images/fold3/images.npy'
MAKS_PATH='/Users/piotrwojcik/Downloads/Fold 3/masks/fold3/masks.npy'
OUTPUT_DIR = '/Users/piotrwojcik/Downloads/fold_3_448/positive/'
DATA_DIR = '/Users/piotrwojcik/Downloads/PN_F3/'
FINAL_TEST_OUTPUT_DIR = '/Users/piotrwojcik/Downloads/pannuke_ftest3/positive'
IDX_DIR = 'pn_fold3_idx'

ratio = 448 / 256


def create_bboxes_for_mask2(inst_map, obj_ids, idx_c):
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
    if len(bboxes) == 0:
        return None
    bboxes, classes = zip(*bboxes)
    boxes = torch.cat(bboxes, dim=0)
    return boxes, list(classes)


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


def generate_pickles(masks, images):
    for m in range(masks.shape[0]):
        mask = masks[m]
        img = images[m]
        img = img.float()
        img = img / 255
        img = F.resize(img, size = (448, 448))
        boxes, classes = create_bboxes_for_mask(mask)
        if boxes is None:
            continue
        img = T.ToPILImage()(img)

        with open(os.path.join(OUTPUT_DIR, f"img_{m}.pkl"), 'wb') as outf:
            pickle.dump(boxes, outf)
        with open(os.path.join(OUTPUT_DIR, f"img_{m}_cls.pkl"), 'wb') as outf:
            pickle.dump(classes.int(), outf)
        img.save(os.path.join(OUTPUT_DIR, f"img_{m}.png"))


def generate_pickles2(data_dir=DATA_DIR, mode='FTest'):
    img_path = os.path.join(data_dir, mode, 'Images')
    mask_path = os.path.join(data_dir, 'mat')
    idx_path = os.path.join(data_dir, IDX_DIR)

    get_files = lambda path: [f for f in listdir(path) if isfile(join(path, f))]

    images = get_files(img_path)
    images.sort()

    masks_files = get_files(mask_path)
    masks_files.sort()

    for m_id, mask_file in enumerate(masks_files):

        mat = scipy.io.loadmat(os.path.join(mask_path, mask_file))
        inst_map = mat['inst_map']
        inst_map = torch.tensor(inst_map)

        idx_f = mask_file.replace('.mat', '.pkl')
        if not Path(os.path.join(idx_path, idx_f)).exists():
            #print(idx_f)
            continue

        idx_cf = mask_file.replace('.mat', '_cls.pkl')

        idx = np.load(os.path.join(idx_path, idx_f), allow_pickle=True)
        idx_c = np.load(os.path.join(idx_path, idx_cf), allow_pickle=True)

        obj_ids = torch.unique(inst_map)[1:]
        dupa = create_bboxes_for_mask2(inst_map, obj_ids[idx], idx_c)
        if dupa is None:
            continue
        boxes, classes = dupa

        file = mask_file.strip('.mat')
        with open(os.path.join(DATA_DIR, mode, 'Pickles', f"{file}.pkl"), 'wb') as outf:
            pickle.dump(boxes, outf)
        with open(os.path.join(DATA_DIR, mode, 'Pickles', f"{file}_cls.pkl"), 'wb') as outf:
            pickle.dump(classes, outf)

    return images


def generate_validation_set(images, data_dir=DATA_DIR, output_dir=FINAL_TEST_OUTPUT_DIR, mode='FTest'):
    img_path = os.path.join(data_dir, mode, 'Images')

    total_boxes = 0

    for img_file in images:
        file = img_file.strip('.png')
        if not Path(os.path.join(data_dir, mode, 'Pickles', f"{file}.pkl")).exists():
            #print(idx_f)
            continue
        boxes = np.load(os.path.join(data_dir, mode, 'Pickles', f"{file}.pkl"), allow_pickle=True)
        classes = np.load(os.path.join(data_dir, mode, 'Pickles', f"{file}_cls.pkl"), allow_pickle=True)
        classes = torch.tensor(classes).int()
        classes = classes - 1
        img = read_image(os.path.join(img_path, img_file))
        img_size = 448
        img = F.resize(img, size = (img_size, img_size))
        img = T.ToPILImage()(img)
        boxes_num = boxes.shape[0]

        boxes = boxes.float()
        boxes[:, 0::2] *= ratio
        boxes[:, 1::2] *= ratio
        boxes = boxes.int()
        for i in range(boxes.shape[0]):
            boxes[i, 2] -= int(boxes[i, 2] == img_size)
            boxes[i, 3] -= int(boxes[i, 3] == img_size)



        with open(os.path.join(output_dir, f"{file}.pkl"), 'wb') as outf:
            pickle.dump(boxes, outf)
        with open(os.path.join(output_dir, f"{file}_cls.pkl"), 'wb') as outf:
            pickle.dump(classes, outf)
        img.save(os.path.join(output_dir, f"{file}.png"))

        total_boxes += boxes_num

    print(total_boxes)


if __name__ == '__main__':
    images_ftest = generate_pickles2(data_dir=DATA_DIR, mode='FTest')
    generate_validation_set(images_ftest, data_dir=DATA_DIR, output_dir=FINAL_TEST_OUTPUT_DIR, mode='FTest')

    #images = torch.tensor(np.load(IMG_PATH, allow_pickle=True))
    #images = images.permute(0, 3, 1, 2)
    #images = images.to(torch.uint8)
    #masks = torch.tensor(np.load(MAKS_PATH, allow_pickle=True))
    #masks = masks.permute(0, 3, 1, 2)
    #print('Start')
    #generate_pickles(masks, images)