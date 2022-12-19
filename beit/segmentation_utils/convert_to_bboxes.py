import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from os import listdir
import pickle
from os.path import isfile, join

import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes, nms, box_convert
from tqdm import tqdm

ASSETS_DIRECTORY = "/projects/ag-bozek/hnaji/code/outp/"
OUTPUT_DIRECTORY = "/scratch/pwojcik/images_ihc/positive/"

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


def expand_bounding_box(bbox, margin, img_shape):
    x_fill = 2 * margin
    x_fill -= min(margin, bbox[0])
    xmin = max(bbox[0] - margin, 0)
    xmax = min(bbox[2] + x_fill, img_shape[0] - 1)

    y_fill = 2 * margin
    y_fill -= min(margin, bbox[2])
    ymin = max(bbox[1] - margin, 0)
    ymax = min(bbox[3] + y_fill, img_shape[1] - 1)

    if (0 in bbox) or (xmax >= img_shape[0] - 1) or (ymax >= img_shape[1] - 1):
        return None

    return box_convert(torch.tensor([xmin, ymin, xmax, ymax]), 'xyxy', 'xyxy').unsqueeze(0)


def create_bboxes_for_image(masks, img_shape=(448, 448)):
    boxes = masks_to_boxes(masks)
    boxes_u = nms(boxes, torch.ones(boxes.shape[0], dtype=torch.float), 0.50)
    boxes_id = boxes[boxes_u]
    boxes_expanded = []
    for i in range(boxes_id.shape[0]):
        expanded = expand_bounding_box(boxes_id[i].numpy(), margin=10, img_shape=img_shape)
        if expanded is not None:
            boxes_expanded.append(expanded)
    return torch.cat(boxes_expanded, dim=0)


if __name__ == '__main__':
    files = [f for f in listdir(ASSETS_DIRECTORY) if isfile(join(ASSETS_DIRECTORY, f))]
    for f in files:
        cat = f.split('_')[-2]
        num = f.split('_')[-1].strip('.pkl')

        if int(num) < 1000:
            continue
        if cat == 'filename':
            filenames = np.load(os.path.join(ASSETS_DIRECTORY, f), allow_pickle=True)
            print(f"Processing {f}...")
            masks = np.load(os.path.join(ASSETS_DIRECTORY, f"seg_{num}.pkl"), allow_pickle=True)
            print(f"Loaded masks.")
            for id in tqdm(range(len(filenames))):
                img = filenames[id]
                mask = torch.tensor(masks[id])
                _mask = []
                for i in range(400):
                    if not torch.all(mask[i] == False):
                        _mask.append(mask[i].unsqueeze(0))
                bboxes = create_bboxes_for_image(torch.cat(_mask, dim=0))
                with open(os.path.join(OUTPUT_DIRECTORY, f"{img.strip('.png')}.pkl"), 'wb') as outf:
                    pickle.dump(bboxes, outf)
