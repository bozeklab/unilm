import argparse

import torch

from pathlib import Path
import utils
from beit.datasets import build_beit_inference_dataset
from beit.run_beit_pretraining import get_model


def get_args():
    parser = argparse.ArgumentParser('BEiT inference script', add_help=False)

    # Model parameters
    parser.add_argument('--model', default=' beit_base_patch16_448_8k_vocab', type=str, metavar='MODEL',
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


def flatten_list(nested_list):
    flattened_list = []
    for element in nested_list:
        if isinstance(element, list):
            flattened_list.extend(flatten_list(element))
        else:
            flattened_list.append(element)
    return flattened_list


@torch.no_grad()
def infere(model, dataset, device):
    pass


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

    infere(model, dataset_train, device)


if __name__ == '__main__':
    opts = get_args()

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)