"""Utilities for loading torchvision ViT-B/16 weights into a timm ViT."""

import os
import torch


def load_torchvision_vit_into_timm(model, weight_path: str):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"ImageNet ViT-B/16 weights not found: {weight_path}")

    source = torch.load(weight_path, map_location='cpu', weights_only=True)
    mapped = {
        'cls_token': source['class_token'],
        'pos_embed': source['encoder.pos_embedding'],
        'patch_embed.proj.weight': source['conv_proj.weight'],
        'patch_embed.proj.bias': source['conv_proj.bias'],
        'norm.weight': source['encoder.ln.weight'],
        'norm.bias': source['encoder.ln.bias'],
        'head.weight': source['heads.head.weight'],
        'head.bias': source['heads.head.bias'],
    }
    pairs = {
        'ln_1.weight': 'norm1.weight',
        'ln_1.bias': 'norm1.bias',
        'self_attention.in_proj_weight': 'attn.qkv.weight',
        'self_attention.in_proj_bias': 'attn.qkv.bias',
        'self_attention.out_proj.weight': 'attn.proj.weight',
        'self_attention.out_proj.bias': 'attn.proj.bias',
        'ln_2.weight': 'norm2.weight',
        'ln_2.bias': 'norm2.bias',
        'mlp.linear_1.weight': 'mlp.fc1.weight',
        'mlp.linear_1.bias': 'mlp.fc1.bias',
        'mlp.linear_2.weight': 'mlp.fc2.weight',
        'mlp.linear_2.bias': 'mlp.fc2.bias',
    }
    for i in range(12):
        source_prefix = f'encoder.layers.encoder_layer_{i}.'
        target_prefix = f'blocks.{i}.'
        for source_key, target_key in pairs.items():
            mapped[target_prefix + target_key] = source[source_prefix + source_key]

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(
        f"[ImageNetLoader] Loaded {len(mapped)} torchvision keys "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
