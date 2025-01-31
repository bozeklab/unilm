import math
import torch
import torch.nn as nn
from functools import partial

from torchvision.ops import roi_align

from beit.positional_encoding import PositionalEncoding
from modeling_finetune import Block, _cfg, PatchEmbed, RelativePositionBias, Mlp
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
import torch.nn.functional as F


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class VisionInstaformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, patch_embed_size=3, instance_size=32, in_chans=3, num_classes=4,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed_size = patch_embed_size
        self.patch_size = patch_size
        self.instance_size = instance_size
        self.img_size = img_size

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.instance_embed = PatchEmbed(img_size=patch_embed_size, patch_size=patch_embed_size,
                                         in_chans=embed_dim, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

            self.pos_embed_x = PositionalEncoding(embed_dim=embed_dim // 4, drop_rate=0., max_len=img_size)
            self.pos_embed_y = PositionalEncoding(embed_dim=embed_dim // 4, drop_rate=0., max_len=img_size)
            self.pos_embed_h = PositionalEncoding(embed_dim=embed_dim // 4, drop_rate=0., max_len=img_size)
            self.pos_embed_w = PositionalEncoding(embed_dim=embed_dim // 4, drop_rate=0., max_len=img_size)

            pos_embed = torch.cat((
                self.pos_embed_x()[..., None, :].repeat(1, 1, img_size, 1),  #
                self.pos_embed_y()[:, None].repeat(1, img_size, 1, 1),  #
                self.pos_embed_w()[:, (patch_size - 1)][:, None, None].repeat(1, img_size, img_size, 1),
                self.pos_embed_h()[:, (patch_size - 1)][:, None, None].repeat(1, img_size, img_size, 1),
            ), dim=3)
            pos_embed = F.interpolate(pos_embed.permute(0, 3, 1, 2),
                                      size=(img_size // patch_size, img_size // patch_size),
                                      mode='bilinear').flatten(2).transpose(-1, -2)
            self.register_buffer('pos_embed', pos_embed)

        else:
            self.pos_embed = None
            self.cls_pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std

        self.head = nn.Linear(embed_dim, num_classes)

        if self.cls_pos_embed is not None:
            trunc_normal_(self.cls_pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def extract_box_feature(self, x, boxes_info, scale_factor):
        h, w = self.patch_embed.patch_shape
        num_box = boxes_info.shape[1]
        batch_size = x.shape[0]
        x = x.view(batch_size, h, w, self.embed_dim).permute(0, 3, 1, 2)
        batch_index = torch.arange(0.0, batch_size).repeat(num_box).view(num_box, -1) \
            .transpose(0, 1).flatten(0, 1).to(x.device)
        roi_box_info = boxes_info.view(-1, 4).to(x.device)

        roi_info = torch.stack((batch_index, roi_box_info[:, 0],
                                roi_box_info[:, 1], roi_box_info[:, 2],
                                roi_box_info[:, 3]), dim=1).to(x.device)
        aligned_out = roi_align(input=x, boxes=roi_info, spatial_scale=scale_factor,
                                output_size=(self.patch_embed_size, self.patch_embed_size))

        aligned_out.view(batch_size, num_box, self.embed_dim, self.patch_embed_size, self.patch_embed_size)[
            torch.where(boxes_info[:, :, 0] == -1)] = 0
        aligned_out.view(-1, self.embed_dim, self.patch_embed_size, self.patch_embed_size)

        return aligned_out

    def add_box_feature(self, x, boxes_features, boxes_info):
        batch_size = x.shape[0]
        num_box = boxes_info.shape[1]
        boxes_features = self.instance_embed(boxes_features).squeeze().view(batch_size, num_box, -1)

        x_coord = boxes_info[..., 0::2].float().mean(dim=2).long()
        y_coord = boxes_info[..., 1::2].float().mean(dim=2).long()
        w = (boxes_info[..., 2] - boxes_info[..., 0]).long()
        h = (boxes_info[..., 3] - boxes_info[..., 1]).long()

        box_pos_embed = torch.cat((
            self.pos_embed_x()[..., None, :].repeat(1, 1, self.img_size, 1),
            self.pos_embed_y()[:, None].repeat(1, self.img_size, 1, 1),
        ), dim=3).squeeze()

        boxes_features += torch.cat((
            box_pos_embed[y_coord, x_coord], box_pos_embed[h, w]
        ), dim=2)

        added_out = torch.cat((x, boxes_features), dim=1)
        return added_out

    def forward_features(self, x, boxes, attention_mask):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)

        attn_prefix = torch.ones(batch_size, seq_len + 1).bool().to(x.device)
        attention_mask = torch.cat([attn_prefix, attention_mask], dim=1)

        if self.pos_embed is not None:
            pos_embed = torch.cat([self.cls_pos_embed, self.pos_embed], dim=1)
            x = x + pos_embed
        x = self.pos_drop(x)

        boxes_features = self.extract_box_feature(x=x[:, 1:], boxes_info=boxes, scale_factor=1. / self.patch_size)
        aggregated_x = self.add_box_feature(x=x, boxes_features=boxes_features, boxes_info=boxes)

        for blk in self.blocks:
            aggregated_x = blk(aggregated_x, attention_mask=attention_mask[:, None, None, :], rel_pos_bias=None)

        return self.norm(aggregated_x)

    def get_last_selfattention(self, x, boxes, attention_mask):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        attn_prefix = torch.ones(batch_size, seq_len + 1).bool().to(x.device)
        attention_mask = torch.cat([attn_prefix, attention_mask], dim=1)

        if self.pos_embed is not None:
            pos_embed = torch.cat([self.cls_pos_embed, self.pos_embed], dim=1)
            x = x + pos_embed
        x = self.pos_drop(x)

        boxes_features = self.extract_box_feature(x=x[:, 1:], boxes_info=boxes, scale_factor=1. / self.patch_size)
        aggregated_x = self.add_box_feature(x=x, boxes_features=boxes_features, boxes_info=boxes)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                aggregated_x = blk(aggregated_x, attention_mask=attention_mask[:, None, None, :], rel_pos_bias=None)
            else:
                # return attention of the last block
                return blk(aggregated_x, attention_mask=attention_mask[:, None, None, :],
                           rel_pos_bias=None, return_attention=True)

    def forward(self, x, boxes, attention_mask):

        x = self.forward_features(x, boxes=boxes, attention_mask=attention_mask)
        num_boxes = boxes.shape[1]
        _, aggregated_box = x[:, :-num_boxes, :], x[:, -num_boxes:, :]

        #pre_head = self.head_mlp(aggregated_box[attention_mask])

        return self.head(aggregated_box[attention_mask])


@register_model
def beit_instaformer_patch16(pretrained=False, **kwargs):
    model = VisionInstaformer(
        img_size=448, patch_size=16, patch_embed_size=3, instance_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), use_rel_pos_bias=False, **kwargs)
    model.default_cfg = _cfg()
    return model
