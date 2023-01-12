import argparse

from beit.datasets import build_beit_inference_dataset, build_instaformer_pretraining_dataset

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


def get_args():
    parser = argparse.ArgumentParser('BEiT inference ds test', add_help=False)

    parser.add_argument('--data_path', default='/Users/piotrwojcik/data/he/', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')
    parser.add_argument('--input_size', default=448, type=int, help='images input size for backbone')
    parser.add_argument('--second_input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    # Augmentation parameters
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")')

    return parser.parse_args()


def flatten_list(nested_list):
    flattened_list = []
    for element in nested_list:
        if isinstance(element, list):
            flattened_list.extend(flatten_list(element))
        else:
            flattened_list.append(element)
    return flattened_list


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def main(args):
    patch_size = [16, 16]

    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    dataset_train = build_instaformer_pretraining_dataset(args)
    print(len(dataset_train))
    print(dataset_train[0])


if __name__ == '__main__':
    opts = get_args()
    main(opts)