# Adapted from FIMA-Q's quant_layers/conv.py (Apache-2.0).
# See docs/FIMA_Q_LICENSE.txt for the license.

"""FIMA-Q's convolution calibration extended to native 3D convolutions."""

import torch
from torch import nn

from ._calibration import check_search_options, iter_records, validate_records
from .quantizers import UniformQuantizer


class QuantConv3d(nn.Conv3d):
    def __init__(self, source, w_bit=4, a_bit=4, quantize_activation=True):
        super().__init__(source.in_channels, source.out_channels, source.kernel_size,
                         source.stride, source.padding, source.dilation, source.groups,
                         source.bias is not None, source.padding_mode)
        self.w_quantizer = UniformQuantizer(w_bit, symmetric=False, channel_wise=True)
        self.a_quantizer = UniformQuantizer(a_bit, symmetric=True)
        self.w_quantizer.scale = nn.Parameter(torch.zeros(source.out_channels, 1, 1, 1, 1))
        self.w_quantizer.zero_point = nn.Parameter(torch.zeros_like(self.w_quantizer.scale))
        self.a_quantizer.scale = nn.Parameter(torch.zeros(1, 1, 1, 1, 1))
        self.to(device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if self.bias is not None:
                self.bias.copy_(source.bias)
        self.mode = 'raw'
        self.calibrated = False
        self.quantize_activation = quantize_activation

    def weight_view(self):
        # AdaRound's elementwise math broadcasts across the kernel dimensions.
        return self.weight

    def forward(self, x):
        if self.mode == 'raw':
            return self._conv_forward(x, self.weight, self.bias)
        if self.mode == 'quant_forward' and self.calibrated:
            return self._conv_forward(self.quant_input(x), self.w_quantizer(self.weight), self.bias)
        raise RuntimeError(f'Conv3d cannot execute in mode {self.mode!r} before calibration')

    def quant_input(self, x):
        return self.a_quantizer(x) if self.quantize_activation else x

    def _weight_bounds(self, count):
        weights = self.weight.flatten(1)
        if weights.dtype not in (torch.float32, torch.float64):
            weights = weights.float()
        pct = torch.linspace(0, 1, count, device=weights.device, dtype=weights.dtype)
        pct = torch.cat((0.99 + (0.9999 - 0.99) * pct.sqrt(), pct.new_ones(1)))
        return (torch.quantile(weights, pct, dim=-1).to(self.weight),
                torch.quantile(weights, 1 - pct, dim=-1).to(self.weight))

    def _weight_outputs(self, x, weights):
        candidates = weights.shape[0]
        per_group = self.out_channels // self.groups
        grouped = weights.reshape(candidates, self.groups, per_group, *self.weight.shape[1:])
        kernels = grouped.transpose(0, 1).reshape(-1, *self.weight.shape[1:])
        bias = None
        if self.bias is not None:
            bias = self.bias.reshape(self.groups, 1, per_group).expand(-1, candidates, -1).reshape(-1)
        output = self._conv_forward(x, kernels, bias)
        batch, _, depth, height, width = output.shape
        grouped_output = output.reshape(batch, self.groups, candidates, per_group, depth, height, width)
        return grouped_output.transpose(1, 2).reshape(batch, candidates, self.out_channels, depth, height, width)

    def _search_weight(self, records, upper, lower, count, chunk):
        quantizer = self.w_quantizer
        levels = 2 * quantizer.n_levels - 1
        scales = (upper - lower) / levels
        scales = torch.where(scales > 0, scales, torch.full_like(scales, torch.finfo(scales.dtype).eps))
        zeros = -(lower / scales).round()
        scores = self.weight.new_zeros((count, self.out_channels))
        for (x,), target in iter_records(records, 1):
            x = self.quant_input(x.to(self.weight))
            target = target.to(self.weight).unsqueeze(1)
            for start in range(0, count, chunk):
                stop = min(start + chunk, count)
                scale = scales[start:stop].reshape(-1, self.out_channels, 1, 1, 1, 1)
                zero = zeros[start:stop].reshape_as(scale)
                codes = (torch.round(self.weight.unsqueeze(0) / scale) + zero).clamp(0, levels)
                output = self._weight_outputs(x, (codes - zero) * scale)
                scores[start:stop] -= (output - target).square().mean(dim=(3, 4, 5)).sum(dim=0)
        best = scores.argmax(dim=0, keepdim=True)
        quantizer.scale.copy_(scales.gather(0, best).reshape_as(quantizer.scale))
        quantizer.zero_point.copy_(zeros.gather(0, best).reshape_as(quantizer.zero_point))
        quantizer.inited = True
        return best

    def _search_input(self, records, scales, count, chunk):
        quantizer = self.a_quantizer
        weights = self.w_quantizer(self.weight)
        scores = self.weight.new_zeros(count)
        for (x,), target in iter_records(records, 1):
            x = x.to(self.weight).unsqueeze(0)
            target = target.to(self.weight).unsqueeze(0)
            for start in range(0, count, chunk):
                stop = min(start + chunk, count)
                scale = scales[start:stop].reshape(-1, 1, 1, 1, 1, 1)
                codes = torch.round(x / scale).clamp(-quantizer.n_levels, quantizer.n_levels - 1)
                output = self._conv_forward((codes * scale).flatten(0, 1), weights, self.bias)
                output = output.reshape(stop - start, *target.shape[1:])
                scores[start:stop] -= (output - target).square().mean(dim=(2, 3, 4, 5)).sum(dim=1)
        quantizer.scale.fill_(scales[scores.argmax()])

    @torch.no_grad()
    def calibrate(self, records, *, eq_n, search_round, candidate_chunk, prev_layer=None):
        if prev_layer is not None:
            raise ValueError('Conv3d has no LayerNorm folding step')
        check_search_options(eq_n, search_round, candidate_chunk)
        if eq_n < 2:
            raise ValueError('Conv3d requires at least two calibration candidates')
        validate_records(records, 1)
        absmax = max(x.abs().amax().item() for (x,), _ in iter_records(records, 1))
        initial = absmax / (self.a_quantizer.n_levels - 0.5)
        self.a_quantizer.scale.fill_(initial if initial > 0 else torch.finfo(self.weight.dtype).eps)
        self.a_quantizer.inited = True
        input_scales = torch.linspace(0.01, 1.2, eq_n + 1, device=self.weight.device,
                                      dtype=self.weight.dtype) * self.a_quantizer.scale.reshape(())
        upper, lower = self._weight_bounds(eq_n)
        best = self._search_weight(records, upper, lower, eq_n, candidate_chunk)
        for _ in range(search_round):
            best = self._search_weight(records, upper.gather(0, best), lower, eq_n, candidate_chunk)
            best = self._search_weight(records, upper, lower.gather(0, best), eq_n, candidate_chunk)
            if self.a_quantizer.n_bits >= 8 or not self.quantize_activation:
                break
            self._search_input(records, input_scales, eq_n, candidate_chunk)
        self.calibrated = True
