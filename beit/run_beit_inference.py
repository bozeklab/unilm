import argparse

import torch

from pathlib import Path
import utils
import pickle
from beit.datasets import build_beit_inference_dataset
from beit.run_beit_pretraining import get_model
from torchvision.ops import roi_align
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

from torch import nn


def get_args():
    parser = argparse.ArgumentParser('BEiT inference script', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_384', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=448, type=int, help='images input size')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--checkpoint', default='', help='use this checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--data_path', default='/Users/piotrwojcik/data/he/', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')
    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')

    return parser.parse_args()


def _flatten_list(nested_list):
    flattened_list = []
    for element in nested_list:
        if isinstance(element, list):
            flattened_list.extend(_flatten_list(element))
        else:
            flattened_list.append(element)
    return flattened_list


tumor_categories = ['plasma cell', 'eosinophil', 'macrophage', 'vessel', 'apoptotic bodies', 'epithelial cell',
                    'normal small lymphocyte', 'large leucocyte', 'stroma', 'immune cells', 'unknown', 'erythrocyte', 'mitose', 'positive', 'tumor']


@torch.no_grad()
def infere(model, dataset, patch_size, device):
    embeddings = []
    labels = []
    images = []

    model.eval()
    for i in range(len(dataset)):
        sample, _ = dataset[i]
        sample = _flatten_list(sample)
        img, nonnormalized_img, bool_masked_pos, boxes_and_labels = sample
        boxes, classes = boxes_and_labels

        img = F.resize(img, size=384)

        img = img.to(device, non_blocking=True).unsqueeze(0)
        boxes = boxes.to(device, non_blocking=True).float()
        bool_masked_pos = torch.tensor(bool_masked_pos)
        bool_masked_pos = torch.zeros_like(bool_masked_pos).bool().to(device, non_blocking=True).unsqueeze(0)
        bool_masked_pos = bool_masked_pos.flatten(1)

        with torch.cuda.amp.autocast():
            x = model.forward_features(x=img, bool_masked_pos=bool_masked_pos)
            x = x[:, 1:]
            batch_size, seq_len, C = x.shape
            x = x.view(batch_size, img.shape[2] // patch_size[0], img.shape[3] // patch_size[1], C)
        aligned_boxes = roi_align(input=x.permute(0, 3, 1, 2), spatial_scale=0.0625, boxes=[boxes], output_size=(3, 3))
        m = nn.AvgPool2d(3, stride=1)
        aligned_boxes = m(aligned_boxes).squeeze()

        aligned_boxes = aligned_boxes.cpu()
        boxes = boxes.cpu()

        for i in range(aligned_boxes.shape[0]):
            embeddings.append(aligned_boxes[i].numpy())
            label = classes[0][i]
            labels.append(label)
            box = boxes[i].numpy().tolist()
            crop = nonnormalized_img[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            images.append(crop.permute(1, 2, 0).numpy())

    return embeddings, labels, images


def main(args):
    print(args)

    device = torch.device(args.device)
    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        print("Load ckpt from %s" % args.checkpoint)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    patch_size = [16, 16]

    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    dataset_train = build_beit_inference_dataset(args)
    print(f"Length of dataset == {len(dataset_train)}")

    embeddings, labels, images = infere(model, dataset_train, patch_size, device)
    output_dict = {'embeddings': embeddings, 'labels': labels, 'images': images}
    with open('outputs/tumor.pickle', 'wb') as f:
       pickle.dump(output_dict, f)


if __name__ == '__main__':
    opts = get_args()

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)