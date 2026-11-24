from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
from timm.models import create_model

import modeling_finetune

from ..adapter import QuantizedBackbone, QuantTarget, ReconstructionUnit
from ..conv3d import QuantConv3d
from ..linear import QuantLinear
from ..matmul import QuantMatMul


class VideoMAEQuantAttention(nn.Module):
    def __init__(self, source, policy):
        super().__init__()
        self.num_heads = source.num_heads
        self.scale = source.scale
        qkv = source.qkv
        if source.q_bias is not None:
            bias = torch.cat((source.q_bias, torch.zeros_like(source.v_bias), source.v_bias))
            qkv.bias = nn.Parameter(bias.detach())
        self.qkv = QuantLinear(qkv, n_groups=3, token_channel_wise=policy.token_channel_wise)
        self.matmul1 = QuantMatMul(self.num_heads, token_channel_wise=policy.token_channel_wise,
                                   head_channel_wise=policy.head_channel_wise)
        self.attn_drop = source.attn_drop
        self.matmul2 = QuantMatMul(self.num_heads, token_channel_wise=policy.token_channel_wise,
                                   head_channel_wise=policy.head_channel_wise)
        self.proj = QuantLinear(source.proj, token_channel_wise=policy.token_channel_wise)
        self.proj_drop = source.proj_drop

    def forward(self, x):
        batch_size, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, tokens, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = self.matmul1(q * self.scale, k.transpose(-2, -1))
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = self.matmul2(attn, v).transpose(1, 2).reshape(batch_size, tokens, -1)
        return self.proj_drop(self.proj(x))


def adapt_videomae(reference, policy):
    if not isinstance(reference, modeling_finetune.VisionTransformer):
        raise TypeError('The VideoMAE adapter expects the local finetuned VisionTransformer')
    model = deepcopy(reference)
    model.use_checkpoint = False
    targets, units = [], []
    if policy.stem != 'float':
        model.patch_embed.proj = QuantConv3d(
            model.patch_embed.proj, a_bit=4 if policy.stem == 'a4' else 8,
            quantize_activation=policy.stem != 'reference',
        )
        targets.append(QuantTarget('patch_embed.proj', model.patch_embed.proj))
        units.append(ReconstructionUnit('patch_embed', model.patch_embed))

    for index, block in enumerate(model.blocks):
        name = f'blocks.{index}'
        block.attn = VideoMAEQuantAttention(block.attn, policy)
        block.mlp.fc1 = QuantLinear(block.mlp.fc1, token_channel_wise=policy.token_channel_wise)
        block.mlp.fc2 = QuantLinear(block.mlp.fc2, token_channel_wise=policy.token_channel_wise)
        norm1 = block.norm1 if policy.fold_layer_norm else None
        norm2 = block.norm2 if policy.fold_layer_norm else None
        targets.extend((
            QuantTarget(f'{name}.attn.qkv', block.attn.qkv, norm1),
            QuantTarget(f'{name}.attn.matmul1', block.attn.matmul1),
            QuantTarget(f'{name}.attn.matmul2', block.attn.matmul2),
            QuantTarget(f'{name}.attn.proj', block.attn.proj),
            QuantTarget(f'{name}.mlp.fc1', block.mlp.fc1, norm2),
            QuantTarget(f'{name}.mlp.fc2', block.mlp.fc2),
        ))
        units.append(ReconstructionUnit(name, block))

    model.head = QuantLinear(model.head, token_channel_wise=policy.token_channel_wise)
    targets.append(QuantTarget('head', model.head))
    units.append(ReconstructionUnit('head', model.head, is_head=True))
    model.to(next(reference.parameters()).device)
    model.eval()
    return QuantizedBackbone(model, policy, tuple(targets), tuple(units))


def build_reference(args):
    model_names = (
        'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_base_patch16_384',
        'vit_large_patch16_224', 'vit_large_patch16_384', 'vit_large_patch16_512',
        'vit_huge_patch16_224',
    )
    if args.model not in model_names:
        raise ValueError(f'Unsupported VideoMAE model {args.model!r}')
    model = create_model(
        args.model, pretrained=False, num_classes=args.nb_classes,
        all_frames=args.num_frames, tubelet_size=args.tubelet_size,
        use_mean_pooling=not args.use_cls, drop_path_rate=args.drop_path,
    )
    if tuple(model.patch_embed.img_size) != (args.input_size, args.input_size):
        raise ValueError('Input resolution does not match the model preset')

    checkpoint = torch.load(args.finetune, map_location='cpu')
    state = next((checkpoint[key] for key in args.model_key.split('|') if key in checkpoint), checkpoint)
    flattened = OrderedDict()
    for key, value in state.items():
        for prefix in ('backbone.', 'encoder.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key in flattened:
            raise ValueError(f'Duplicate checkpoint weight {key!r}')
        flattened[key] = value
    model.load_state_dict(flattened, strict=True)
    return model.eval()
