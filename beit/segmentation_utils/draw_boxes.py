import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes, nms, box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

from skimage.measure import label, regionprops

import cv2


plt.rcParams["savefig.bbox"] = "tight"


def expand_bounding_box(bbox, margin, img_shape):
    x_fill = 2 * margin
    x_fill -= min(margin, bbox[0])
    xmin = max(bbox[0] - margin, 0)
    xmax = min(bbox[2] + x_fill, img_shape[0] - 1)

    y_fill = 2 * margin
    y_fill -= min(margin, bbox[2])
    ymin = max(bbox[1] - margin, 0)
    ymax = min(bbox[3] + y_fill, img_shape[1] - 1)

    if (0.0 in bbox) or (xmax >= img_shape[0] - 1) or (ymax >= img_shape[1] - 1):
        return None

    return box_convert(torch.tensor([xmin, ymin, xmax, ymax]), 'xyxy', 'xyxy').unsqueeze(0)


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
    img_path = '/Users/piotrwojcik/Downloads/wsi_001-tile-r101bis-c69'
    img = read_image(img_path + '.png')

    mask = np.load(img_path + '_mask.pkl', allow_pickle=True)
    mask_sum = mask.int().sum(dim=0)
    mask_sum = (mask_sum > 0).unsqueeze(0)
    mask_sum = torch.permute(mask_sum, [1, 2, 0])
    lbl_0 = label(mask_sum)
    props = regionprops(lbl_0)
    bboxes = []
    for prop in props:
        print('Found bbox', prop.bbox)
        bboxes.append(prop.bbox)

    img_0 = cv2.imread(img_path + '.png')
    img_1 = img_0.copy()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    for bbox in bboxes:
        cv2.rectangle(img_1, (bbox[1], bbox[0]), (bbox[4], bbox[3]), (255, 0, 0), 2)
    ax1.imshow(img_0)
    ax2.imshow(mask_sum[..., 0], cmap='gray')
    ax3.imshow(img_1)
    plt.show()
    exit()


    drawn_masks = draw_segmentation_masks(img, mask_sum, alpha=0.8, colors="blue")
    show(drawn_masks)
    boxes = masks_to_boxes(mask_sum)
    print(boxes)
    #drawn_boxes = draw_bounding_boxes(img, boxes, colors="red")
    #show(drawn_boxes)
    plt.show()
    #print(bboxes)

    scores = []
    for id, i in enumerate(mask):
        score = (i == True).sum().numpy().item()
        scores.append(score)
        #print(score)
        drawn_masks = draw_segmentation_masks(img, i, alpha=0.8, colors="blue")
        #print(id)
        #show(drawn_masks)
        #plt.show()

    bboxes = np.load(img_path + '.pkl', allow_pickle=True)
    bboxes_orig = np.load(img_path + '_orig.pkl', allow_pickle=True)
    orig = []

    #drawn_boxes = draw_bounding_boxes(img, bboxes, colors="red")
    #print(bboxes)
    boxes = masks_to_boxes(mask)
    new_scores = []
    for i in range(boxes.shape[0]):
        orig.append(boxes[i].unsqueeze(0))
        new_scores.append(scores[i])
    orig_boxes = torch.cat(orig, dim=0)
    boxes_u = nms(orig_boxes, torch.tensor(new_scores, dtype=torch.float), 0.30)
    boxes_id = orig_boxes[boxes_u]
    drawn_boxes_orig = draw_bounding_boxes(img, boxes, colors="blue")
    #show(drawn_boxes_orig)
    boxes_expanded = []
    for i in range(boxes_id.shape[0]):
        expanded = expand_bounding_box(boxes_id[i].numpy(), margin=10, img_shape=(448, 448))
        if expanded is not None:
            boxes_expanded.append(expanded)
    img = img[:, :449, :449]
    drawn_boxes = draw_bounding_boxes(img, torch.cat(boxes_expanded, dim=0), colors="red")

    show(drawn_boxes)
    plt.show()



