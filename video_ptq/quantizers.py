# Copyright 2025 FIMA-Q authors.
#
# Adapted from FIMA-Q (https://github.com/ShiheWang/FIMA-Q), licensed under
# the Apache License, Version 2.0. See docs/FIMA_Q_LICENSE.txt.

"""Uniform fake quantization and AdaRound used by the video PTQ operators."""

import torch
from torch import nn


def round_ste(x):
    """Round in the forward pass and use the identity gradient."""
    return (x.round() - x).detach() + x


def floor_ste(x):
    """Floor in the forward pass and use the identity gradient."""
    return (x.floor() - x).detach() + x


def ceil_ste(x):
    """Ceil in the forward pass and use the identity gradient."""
    return (x.ceil() - x).detach() + x


class UniformQuantizer(nn.Module):
    """FIMA-Q's signed or affine uniform fake quantizer."""

    def __init__(self, n_bits=8, symmetric=False, channel_wise=False):
        super().__init__()
        if n_bits < 1 or n_bits > 32:
            raise ValueError("n_bits must be between 1 and 32")
        self.sym = bool(symmetric)
        self.n_bits = int(n_bits)
        self.n_levels = 2 ** (self.n_bits - 1)
        self.channel_wise = bool(channel_wise)
        self.drop_prob = 1.0
        self.inited = False
        self.training_mode = False
        self.scale = nn.Parameter(torch.zeros(1))
        if self.sym:
            self.register_parameter("zero_point", None)
        else:
            self.zero_point = nn.Parameter(torch.zeros(1))

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False

    def forward(self, x):
        if self.n_bits == 32:
            return x
        if not self.inited:
            raise RuntimeError("UniformQuantizer must be calibrated before quant_forward")
        if self.training_mode and self.drop_prob < 1.0:
            x_orig = x
        x_int = round_ste(x / self.scale) if self.training_mode else torch.round(x / self.scale)
        if self.sym:
            x_quant = x_int.clamp(-self.n_levels, self.n_levels - 1)
            x_dequant = x_quant * self.scale
        else:
            zero_point = round_ste(self.zero_point) if self.training_mode else torch.round(self.zero_point)
            x_quant = (x_int + zero_point).clamp(0, 2 * self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * self.scale
        if self.training_mode and self.drop_prob < 1.0:
            return torch.where(torch.rand_like(x) < self.drop_prob, x_dequant, x_orig)
        return x_dequant

    def __repr__(self):
        return "{}(n_bits={}, sym={}, channel_wise={})".format(
            self.__class__.__name__, self.n_bits, self.sym, self.channel_wise
        )


class AdaRoundQuantizer(nn.Module):
    """FIMA-Q's learned hard-sigmoid weight rounding quantizer."""

    def __init__(self, uq, weight_tensor, round_mode="learned_hard_sigmoid"):
        super().__init__()
        self.n_bits = uq.n_bits
        self.n_levels = uq.n_levels
        self.channel_wise = uq.channel_wise
        self.sym = uq.sym
        self.inited = uq.inited
        self.training_mode = False
        self.drop_prob = 1.0
        self.scale = nn.Parameter(uq.scale.detach().clone(), requires_grad=False)
        if self.sym:
            self.register_parameter("zero_point", None)
        else:
            self.zero_point = nn.Parameter(uq.zero_point.detach().clone(), requires_grad=False)
        self.round_mode = round_mode
        self.soft_targets = False
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2.0 / 3.0
        self.alpha = None
        self.init_alpha(weight_tensor.detach().clone())

    def init_alpha(self, x):
        if self.round_mode != "learned_hard_sigmoid":
            raise NotImplementedError("AdaRound alpha initialization requires learned_hard_sigmoid")
        x_floor = torch.floor(x / self.scale)
        rest = (x / self.scale) - x_floor
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
        self.alpha = nn.Parameter(alpha)

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def forward(self, x):
        if self.round_mode == "nearest":
            x_int = torch.round(x / self.scale)
        elif self.round_mode == "nearest_ste":
            x_int = round_ste(x / self.scale)
        elif self.round_mode == "learned_hard_sigmoid":
            x_floor = torch.floor(x / self.scale)
            rounding = self.get_soft_targets() if self.soft_targets else (self.alpha >= 0).to(x.dtype)
            x_int = x_floor + rounding
        else:
            raise ValueError("Wrong rounding mode")
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels, self.n_levels - 1)
            return x_quant * self.scale
        x_quant = torch.clamp(x_int + self.zero_point, 0, 2 * self.n_levels - 1)
        return (x_quant - self.zero_point) * self.scale

    def get_hard_value(self, original_weight):
        original_shape = original_weight.shape
        rounded = torch.floor(original_weight.reshape_as(self.alpha) / self.scale)
        rounded = rounded + (self.alpha >= 0).to(original_weight.dtype)
        return (rounded * self.scale).reshape(original_shape)

    def __repr__(self):
        return "{}(n_bits={}, sym={}, channel_wise={}, round_mode={})".format(
            self.__class__.__name__, self.n_bits, self.sym, self.channel_wise, self.round_mode
        )
