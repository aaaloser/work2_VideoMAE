import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import nn


FORMAT_VERSION = 1
QUANTIZERS = ('w_quantizer', 'a_quantizer', 'A_quantizer', 'B_quantizer')


def _target_names(backbone):
    return [target.name for target in backbone.targets]


def save_quantized(path, backbone, *, stage, metadata):
    if stage not in ('calibrated', 'reconstructed'):
        raise ValueError(f'Invalid PTQ checkpoint stage {stage!r}')
    if not all(target.module.calibrated for target in backbone.targets):
        raise ValueError('Cannot save a checkpoint with uncalibrated targets')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=FORMAT_VERSION, stage=stage, policy=asdict(backbone.policy),
                   targets=_target_names(backbone), metadata=metadata,
                   model=backbone.model.state_dict())
    with NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix='.tmp', delete=False) as output:
        temporary_path = Path(output.name)
        try:
            torch.save(payload, output)
            output.flush()
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def load_quantized(path, backbone, *, expected_stage=None):
    payload = torch.load(path, map_location='cpu')
    if payload['version'] != FORMAT_VERSION:
        raise ValueError(f"Unsupported PTQ checkpoint format {payload['version']}")
    stage = payload['stage']
    if stage not in ('calibrated', 'reconstructed') or (expected_stage and stage != expected_stage):
        raise ValueError(f'PTQ checkpoint stage {stage!r} does not match {expected_stage!r}')
    if payload['policy'] != asdict(backbone.policy) or payload['targets'] != _target_names(backbone):
        raise ValueError('PTQ checkpoint operator coverage or policy does not match')

    state = payload['model']
    for target in backbone.targets:
        module = target.module
        bias_key = f'{target.name}.bias'
        if getattr(module, 'bias', None) is None and bias_key in state:
            if target.prev_layer is None:
                raise ValueError(f'Unexpected bias for {target.name}')
            module.bias = nn.Parameter(module.weight.new_empty(state[bias_key].shape))
        for name in QUANTIZERS:
            quantizer = getattr(module, name, None)
            if quantizer is None:
                continue
            for field in ('scale', 'zero_point'):
                key = f'{target.name}.{name}.{field}'
                current = getattr(quantizer, field, None)
                if current is not None and key in state and state[key].shape != current.shape:
                    new_value = current.new_empty(state[key].shape)
                    setattr(quantizer, field, nn.Parameter(new_value, requires_grad=current.requires_grad))

    backbone.model.load_state_dict(state, strict=True)
    for target in backbone.targets:
        module = target.module
        for name in QUANTIZERS:
            quantizer = getattr(module, name, None)
            if quantizer is not None:
                quantizer.inited = True
        module.calibrated = True
        module.mode = 'quant_forward'
    backbone.model.eval()
    return payload['metadata']
