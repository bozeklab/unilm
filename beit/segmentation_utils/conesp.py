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

DATA_DIR = '/Users/piotrwojcik/Downloads/CoNSeP'
IDX_DIR = 'consep_idx'
TEST_OUTPUT_DIR = '/Users/piotrwojcik/Downloads/conesp_test/positive'
FINAL_TEST_OUTPUT_DIR = '/Users/piotrwojcik/Downloads/conesp_ftest2/positive'
TRAIN_OUTPUT_DIR = '/Users/piotrwojcik/Downloads/conesp_train/positive'

class_values = {"other": 1, "inflammatory": 2, "healthy epithelial": 3,
                "dysplastic/malignant epithelial": 4, "fibroblast": 5, "muscle": 6, "endothelial": 7}

CROP_SIZE = 448


def create_bboxes_for_mask(inst_map, type_map, obj_ids):
    bboxes = []
    for value in obj_ids.numpy().tolist():
        mask = type_map == value
        insta_map_v = inst_map * mask.int()
        lbl = label(insta_map_v)
        props = regionprops(lbl)
        for prop in props:
            bbox = prop.bbox

            #area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            #if area == 0:
            #    continue

            b = torch.tensor([bbox[1], bbox[0], bbox[3], bbox[2]]).unsqueeze(0)
            bboxes.append((b, value))

    random.shuffle(bboxes)
    bboxes, classes = zip(*bboxes)
    boxes = torch.cat(bboxes, dim=0)
    return boxes, list(classes)


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
    bboxes, classes = zip(*bboxes)
    boxes = torch.cat(bboxes, dim=0)
    return boxes, list(classes)


def get_bounding_boxes(boxes, classes, x_offset, y_offset):
    idx = []
    for i in range(boxes.shape[0]):
        if (boxes[i, 0] >= y_offset) and (boxes[i, 1] >= x_offset) \
                and (boxes[i, 2] < y_offset + CROP_SIZE) and (boxes[i, 3] < x_offset + CROP_SIZE):
            idx.append(i)

    if len(idx) == 0:
        return None, None, None

    remaining_idx = set(range(boxes.shape[0])).difference(set(idx))

    n_boxes = boxes[idx, ...]
    n_boxes[:, 0::2] -= y_offset
    n_boxes[:, 1::2] -= x_offset

    return n_boxes, classes[idx, ...], list(remaining_idx)


def generate_pickles(data_dir=DATA_DIR, mode='FTest'):
    img_path = os.path.join(data_dir, mode, 'Images')
    mask_path = os.path.join(data_dir, mode, 'Labels')

    get_files = lambda path: [f for f in listdir(path) if isfile(join(path, f))]

    images = get_files(img_path)
    masks_files = get_files(mask_path)

    for mask_file in masks_files:
        mat = scipy.io.loadmat(os.path.join(mask_path, mask_file))
        inst_map = mat['inst_map']
        inst_map = torch.tensor(inst_map)

        type_map = mat['type_map']
        type_map = torch.tensor(type_map)

        obj_ids = torch.unique(torch.tensor(mat['inst_type']))

        boxes, classes = create_bboxes_for_mask(inst_map, torch.tensor(type_map), torch.tensor(obj_ids))

        file = mask_file.strip('.mat')
        #with open(os.path.join(DATA_DIR, mode, 'Pickles', f"t{file}.pkl"), 'wb') as outf:
        #    pickle.dump(boxes, outf)
        #with open(os.path.join(DATA_DIR, mode, 'Pickles', f"t{file}_cls.pkl"), 'wb') as outf:
        #    pickle.dump(classes, outf)

    return images


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
        idx_cf = mask_file.replace('.mat', '_cls.pkl')

        idx = np.load(os.path.join(idx_path, idx_f), allow_pickle=True)
        idx_c = np.load(os.path.join(idx_path, idx_cf), allow_pickle=True)

        obj_ids = torch.unique(inst_map)[1:]
        boxes, classes = create_bboxes_for_mask2(inst_map, obj_ids[idx], idx_c)

        file = mask_file.strip('.mat')
        #with open(os.path.join(DATA_DIR, mode, 'Pickles', f"t{file}.pkl"), 'wb') as outf:
        #    pickle.dump(boxes, outf)
        #with open(os.path.join(DATA_DIR, mode, 'Pickles', f"t{file}_cls.pkl"), 'wb') as outf:
        #    pickle.dump(classes, outf)

    return images


def generate_training_set(images, data_dir=DATA_DIR, output_dir=TRAIN_OUTPUT_DIR):
    mode = 'Train'
    img_path = os.path.join(data_dir, mode, 'Images')

    for img_file in images:
        file = img_file.strip('.png')
        boxes = np.load(os.path.join(data_dir, mode, 'Pickles', f"{file}.pkl"), allow_pickle=True)
        classes = np.load(os.path.join(data_dir, mode, 'Pickles', f"{file}_cls.pkl"), allow_pickle=True)
        classes = torch.tensor(classes).int()
        img = read_image(os.path.join(img_path, img_file))
        offsets = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 551]

        for o_v in offsets:
            for o_h in offsets:
                crop = img[:, o_v:(o_v + CROP_SIZE), o_h:(o_h + CROP_SIZE)]
                crop = T.ToPILImage()(crop)
                #crop_boxes, crop_classes, _ = get_bounding_boxes(boxes, classes, o_v, o_h)
                #if crop_boxes is None:
                #    continue

               # with open(os.path.join(output_dir, f"{file}_aug_{o_v}_{o_h}.pkl"), 'wb') as outf:
               #     pickle.dump(crop_boxes, outf)
               # with open(os.path.join(output_dir, f"{file}_aug_{o_v}_{o_h}_cls.pkl"), 'wb') as outf:
               #     pickle.dump(crop_classes, outf)
               # crop.save(os.path.join(output_dir, f"{file}_aug_{o_v}_{o_h}.png"))

        #generate different scale
        img = read_image(os.path.join(img_path, img_file))
        img_size = 796
        img = F.resize(img, size = (img_size, img_size))
        boxes = np.load(os.path.join(data_dir, mode, 'Pickles', f"{file}.pkl"), allow_pickle=True)
        ratio = img_size / 1000
        boxes = boxes.float()
        boxes[:, 0::2] *= ratio
        boxes[:, 1::2] *= ratio
        boxes = boxes.int()
        for i in range(boxes.shape[0]):
            boxes[i, 2] -= int(boxes[i, 2] == img_size)
            boxes[i, 3] -= int(boxes[i, 3] == img_size)
        offsets = [0, CROP_SIZE - 100]
        for o_v in offsets:
            for o_h in offsets:
                crop = img[:, o_v:(o_v + CROP_SIZE), o_h:(o_h + CROP_SIZE)]
                crop = T.ToPILImage()(crop)
                crop_boxes, crop_classes, _ = get_bounding_boxes(boxes, classes, o_v, o_h)
                if crop_boxes is None:
                   continue

                with open(os.path.join(output_dir, f"{file}_saug_{o_v}_{o_h}.pkl"), 'wb') as outf:
                    pickle.dump(crop_boxes, outf)
                with open(os.path.join(output_dir, f"{file}_saug_{o_v}_{o_h}_cls.pkl"), 'wb') as outf:
                    pickle.dump(crop_classes, outf)
                crop.save(os.path.join(output_dir, f"{file}_saug_{o_v}_{o_h}.png"))


def generate_validation_set(images, data_dir=DATA_DIR, output_dir=TEST_OUTPUT_DIR, mode='FTest'):
    img_path = os.path.join(data_dir, mode, 'Images')

    total_boxes = 0

    for img_file in images:
        file = img_file.strip('.png')
        boxes = np.load(os.path.join(data_dir, mode, 'Pickles', f"{file}.pkl"), allow_pickle=True)
        classes = np.load(os.path.join(data_dir, mode, 'Pickles', f"{file}_cls.pkl"), allow_pickle=True)
        classes = torch.tensor(classes).int()
        img = read_image(os.path.join(img_path, img_file))
        img_size = 796
        img = F.resize(img, size = (img_size, img_size))

        boxes_num = boxes.shape[0]

        ratio = img_size / 1000
        boxes = boxes.float()
        boxes[:, 0::2] *= ratio
        boxes[:, 1::2] *= ratio
        boxes = boxes.int()
        for i in range(boxes.shape[0]):
            boxes[i, 2] -= int(boxes[i, 2] == img_size)
            boxes[i, 3] -= int(boxes[i, 3] == img_size)

        offset = CROP_SIZE - 100

        lu = img[:, 0:CROP_SIZE, 0:CROP_SIZE]
        lu = T.ToPILImage()(lu)
        lu_p, lu_c, remaining_idx = get_bounding_boxes(boxes, classes, 0, 0)
        boxes = boxes[remaining_idx]
        classes = classes[remaining_idx]

        ld = img[:, offset:img_size, 0:CROP_SIZE]
        ld = T.ToPILImage()(ld)
        ld_p, ld_c, remaining_idx = get_bounding_boxes(boxes, classes, offset, 0)
        boxes = boxes[remaining_idx]
        classes = classes[remaining_idx]

        ru = img[:, 0:CROP_SIZE, offset:img_size]
        ru = T.ToPILImage()(ru)
        ru_p, ru_c, remaining_idx = get_bounding_boxes(boxes, classes, 0, offset)
        boxes = boxes[remaining_idx]
        classes = classes[remaining_idx]

        rd = img[:, offset:img_size, offset:img_size]
        rd = T.ToPILImage()(rd)
        rd_p, rd_c, remaining_idx = get_bounding_boxes(boxes, classes, offset, offset)
        #print(boxes[remaining_idx])
        assert(len(remaining_idx) == 0)

        with open(os.path.join(output_dir, f"{file}_aug_0_0.pkl"), 'wb') as outf:
            pickle.dump(lu_p, outf)
        with open(os.path.join(output_dir, f"{file}_aug_0_0_cls.pkl"), 'wb') as outf:
            pickle.dump(lu_c, outf)
        lu.save(os.path.join(output_dir, f"{file}_aug_0_0.png"))

        with open(os.path.join(output_dir, f"{file}_aug_1_0.pkl"), 'wb') as outf:
            pickle.dump(ld_p, outf)
        with open(os.path.join(output_dir, f"{file}_aug_1_0_cls.pkl"), 'wb') as outf:
            pickle.dump(ld_c, outf)
        ld.save(os.path.join(output_dir, f"{file}_aug_1_0.png"))

        with open(os.path.join(output_dir, f"{file}_aug_0_1.pkl"), 'wb') as outf:
            pickle.dump(ru_p, outf)
        with open(os.path.join(output_dir, f"{file}_aug_0_1_cls.pkl"), 'wb') as outf:
            pickle.dump(ru_c, outf)
        ru.save(os.path.join(output_dir, f"{file}_aug_0_1.png"))

        with open(os.path.join(output_dir, f"{file}_aug_1_1.pkl"), 'wb') as outf:
            pickle.dump(rd_p, outf)
        with open(os.path.join(output_dir, f"{file}_aug_1_1_cls.pkl"), 'wb') as outf:
            pickle.dump(rd_c, outf)
        rd.save(os.path.join(output_dir, f"{file}_aug_1_1.png"))

        total_boxes += boxes_num

    print(total_boxes)


if __name__ == '__main__':
    #images_test = generate_pickles(data_dir=DATA_DIR, mode='Test')
    #images_train = generate_pickles(data_dir=DATA_DIR, mode='Train')

    images_ftest = generate_pickles2(data_dir=DATA_DIR, mode='FTest')

    #generate_training_set(images_train, data_dir=DATA_DIR, output_dir=TRAIN_OUTPUT_DIR)
    generate_validation_set(images_ftest, data_dir=DATA_DIR, output_dir=FINAL_TEST_OUTPUT_DIR, mode='FTest')
    #generate_validation_set(images_test, data_dir=DATA_DIR, output_dir=TEST_OUTPUT_DIR, mode='Test')


