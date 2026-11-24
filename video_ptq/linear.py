# Copyright 2025 FIMA-Q authors.
#
# Adapted from FIMA-Q's quant_layers/linear.py (Apache-2.0). See
# docs/FIMA_Q_LICENSE.txt. Candidate scoring follows its asymmetric percentile
# search while replaying CPU records in bounded microbatches.

"""Asymmetric fake-quant Linear with FIMA-Q percentile calibration."""

import torch
from torch import nn
from torch.nn import functional as F

from ._calibration import (
    check_search_options,
    iter_records,
    move_record,
    quantile_summary,
    validate_records,
)
from .quantizers import UniformQuantizer


def _candidate_count(eq_n, n_bits):
    num_zero_points = min(16, 2 ** n_bits)
    if eq_n < num_zero_points or eq_n % num_zero_points:
        raise ValueError(
            "eq_n must be a positive multiple of {} for the FIMA-Q zero-point grid".format(
                num_zero_points
            )
        )
    return num_zero_points, eq_n // num_zero_points


def _assign_parameter(module, name, value):
    setattr(module, name, nn.Parameter(value.detach().clone()))


class _TransformedRecords(object):
    """Replay records after the FIMA-Q upstream affine transformation."""

    def __init__(self, records, ratio, offset):
        self.records = records
        self.root = getattr(records, "root", None)
        self.ratio = ratio.detach().cpu()
        self.offset = offset.detach().cpu()

    def __iter__(self):
        for (x,), raw_output in iter_records(self.records, 1):
            ratio = self.ratio.to(dtype=x.dtype)
            offset = self.offset.to(dtype=x.dtype)
            yield ((x / ratio - offset,), raw_output)


class QuantLinear(nn.Linear):
    """FIMA-Q asymmetric W/A fake quantization for ``nn.Linear``.

    ``n_groups`` divides output channels into logical groups. Each output
    channel has its own weight scale and zero point, represented in the
    original FIMA-Q layout ``[n_groups, out_per_group, 1]``.
    """

    def __init__(self, source, w_bit=4, a_bit=4, n_groups=1, token_channel_wise=True):
        if not isinstance(source, nn.Linear):
            raise TypeError("source must be nn.Linear")
        if n_groups < 1 or source.out_features % n_groups:
            raise ValueError("n_groups must divide source.out_features")
        super().__init__(source.in_features, source.out_features, source.bias is not None)
        self.to(device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if self.bias is not None:
                self.bias.copy_(source.bias)
        self.weight.requires_grad_(source.weight.requires_grad)
        if self.bias is not None:
            self.bias.requires_grad_(source.bias.requires_grad)

        self.mode = "raw"
        self.w_bit = int(w_bit)
        self.a_bit = int(a_bit)
        self.n_groups = int(n_groups)
        self.out_per_group = self.out_features // self.n_groups
        self.token_channel_wise = bool(token_channel_wise)
        self.w_quantizer = UniformQuantizer(w_bit, symmetric=False, channel_wise=True)
        self.a_quantizer = UniformQuantizer(a_bit, symmetric=False, channel_wise=False)
        _assign_parameter(
            self.w_quantizer,
            "scale",
            torch.zeros(self.n_groups, self.out_per_group, 1, device=self.weight.device, dtype=self.weight.dtype),
        )
        _assign_parameter(
            self.w_quantizer,
            "zero_point",
            torch.zeros(self.n_groups, self.out_per_group, 1, device=self.weight.device, dtype=self.weight.dtype),
        )
        _assign_parameter(self.a_quantizer, "scale", torch.zeros(1, device=self.weight.device, dtype=self.weight.dtype))
        _assign_parameter(
            self.a_quantizer, "zero_point", torch.zeros(1, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.calibrated = False

    @property
    def device(self):
        return self.weight.device

    def weight_view(self):
        """Return the logical grouped weight used by FIMA-Q/AdaRound."""
        return self.weight.reshape(self.n_groups, self.out_per_group, self.in_features)

    def forward(self, x):
        if self.mode == "raw":
            return F.linear(x, self.weight, self.bias)
        if self.mode == "quant_forward":
            if not self.calibrated:
                raise RuntimeError("Linear must be calibrated before quant_forward")
            return F.linear(self.quant_input(x), self.quant_weight(), self.bias)
        raise NotImplementedError("unsupported QuantLinear mode: {}".format(self.mode))

    def quant_input(self, x):
        return self.a_quantizer(x)

    def quant_weight(self):
        return self.w_quantizer(self.weight_view()).reshape_as(self.weight)

    def quant_weight_bias(self):
        return self.quant_weight(), self.bias

    @torch.no_grad()
    def calibrate(self, records, *, eq_n, search_round, candidate_chunk, prev_layer=None):
        check_search_options(eq_n, search_round, candidate_chunk)
        validate_records(records, 1)
        self._validate_candidate_grid(eq_n)
        if prev_layer is not None:
            records = self._affine_reparameterize(records, prev_layer, eq_n, candidate_chunk)

        if self.w_bit != 32:
            weight_candidates = self._weight_candidates(eq_n)
        else:
            weight_candidates = None

        if self.w_bit == 32:
            self.w_quantizer.inited = True
        else:
            self._initialize_weight_quantizer_self(weight_candidates, candidate_chunk)

        if self.a_bit == 32:
            activation_candidates = None
            self.a_quantizer.inited = True
        else:
            activation_candidates = self._activation_candidates(records, eq_n)
            self._initialize_activation_quantizer_self(activation_candidates, records, candidate_chunk)

        for _ in range(int(search_round)):
            if self.w_bit != 32:
                self._search_weight_for_output(weight_candidates, records, candidate_chunk)
            if self.a_bit != 32:
                self._search_activation_for_output(activation_candidates, records, candidate_chunk)

        if self.token_channel_wise:
            sample_inputs, _ = next(iter_records(records, 1))
            x = sample_inputs[0]
            if x.ndim == 3:
                token_count = x.shape[1]
                self.a_quantizer.scale = nn.Parameter(
                    self.a_quantizer.scale.detach().reshape(1, 1, 1).expand(1, token_count, 1).clone()
                )
        self.w_quantizer.inited = True
        self.a_quantizer.inited = True
        self.calibrated = True
        return self

    def _validate_candidate_grid(self, eq_n):
        if self.w_bit != 32:
            _candidate_count(eq_n, self.w_bit)
        if self.a_bit not in (32,):
            _candidate_count(eq_n, self.a_bit)

    def _weight_candidates(self, eq_n):
        num_zero_points, num_scales = _candidate_count(eq_n, self.w_bit)
        q = torch.tensor([0.9, 1.0], dtype=torch.float32)
        grouped = self.weight_view().detach()
        uppers = torch.quantile(grouped, q.to(grouped.device), dim=-1).unsqueeze(-1)
        lowers = torch.quantile(grouped, (1.0 - q).to(grouped.device), dim=-1).unsqueeze(-1)
        low_range = uppers[0:1] - lowers[0:1]
        high_range = uppers[1:2] - lowers[1:2]
        alphas = torch.linspace(0.0, 1.0, steps=num_scales, device=self.device, dtype=self.weight.dtype)
        delta = low_range + alphas.view(-1, 1, 1, 1) * (high_range - low_range)
        scale_base = delta / float(2 * self.w_quantizer.n_levels - 1)
        scale_base = scale_base.clamp_min(torch.finfo(self.weight.dtype).eps)
        scales = scale_base.repeat(num_zero_points, 1, 1, 1)
        zp_values = torch.arange(num_zero_points, device=self.device, dtype=self.weight.dtype)
        zero_points = zp_values.repeat_interleave(num_scales).view(-1, 1, 1, 1)
        zero_points = zero_points.expand(-1, self.n_groups, self.out_per_group, 1).contiguous()
        return scales, zero_points

    def _activation_candidates(self, records, eq_n, grouping="tensor"):
        num_zero_points, num_scales = _candidate_count(eq_n, self.a_bit)
        percentiles = quantile_summary(
            records, 1, 0, [0.9, 1.0, 0.1, 0.0], grouping=grouping
        ).to(device=self.device, dtype=self.weight.dtype)
        uppers, lowers = percentiles[:2], percentiles[2:]
        low_range = uppers[0] - lowers[0]
        high_range = uppers[1] - lowers[1]
        alphas = torch.linspace(0.0, 1.0, steps=num_scales, device=self.device, dtype=self.weight.dtype)
        scales = low_range.unsqueeze(-1) + alphas.unsqueeze(0) * (high_range - low_range).unsqueeze(-1)
        scales = (scales / float(2 * self.a_quantizer.n_levels - 1)).clamp_min(1e-4)
        scales = scales.repeat(1, num_zero_points)
        zero_points = torch.arange(num_zero_points, device=self.device, dtype=self.weight.dtype)
        zero_points = zero_points.repeat_interleave(num_scales).unsqueeze(0).expand(scales.shape[0], -1)
        # FIMA-Q appends a duplicate final candidate; scoring still traverses eq_n.
        scales = torch.cat([scales, scales[:, -1:]], dim=-1)
        zero_points = torch.cat([zero_points, zero_points[:, -1:]], dim=-1)
        return scales, zero_points

    def _initialize_weight_quantizer_self(self, candidates, candidate_chunk):
        scales, zero_points = candidates
        weight = self.weight_view().detach().unsqueeze(0)
        scores = []
        for start in range(0, scales.shape[0], candidate_chunk):
            stop = min(scales.shape[0], start + candidate_chunk)
            cur_scale = scales[start:stop]
            cur_zero = zero_points[start:stop]
            quant = torch.round(weight / cur_scale) + cur_zero
            quant = quant.clamp(0, 2 * self.w_quantizer.n_levels - 1)
            dequant = (quant - cur_zero) * cur_scale
            scores.append(-((weight - dequant) ** 2).mean(dim=-1))
        score = torch.cat(scores, dim=0)
        best = score.argmax(dim=0, keepdim=True)
        scale = torch.gather(scales, 0, best.unsqueeze(-1)).squeeze(0)
        zero = torch.gather(zero_points, 0, best.unsqueeze(-1)).squeeze(0)
        _assign_parameter(self.w_quantizer, "scale", scale)
        _assign_parameter(self.w_quantizer, "zero_point", zero)
        self.w_quantizer.inited = True

    def _initialize_activation_quantizer_self(self, candidates, records, candidate_chunk):
        scales, zero_points = candidates
        groups = scales.shape[0]
        n_candidates = eq_candidate_count(candidates)
        scores = torch.zeros(groups, scales.shape[1], device=self.device, dtype=self.weight.dtype)
        for inputs, _raw_output in iter_records(records, 1):
            x = inputs[0]
            x = x.to(device=self.device, dtype=self.weight.dtype)
            x_expanded = x.unsqueeze(-1)
            for start in range(0, n_candidates, candidate_chunk):
                stop = min(n_candidates, start + candidate_chunk)
                cur_scale = scales[:, start:stop]
                cur_zero = zero_points[:, start:stop]
                quant = (torch.round(x_expanded / cur_scale) + cur_zero).clamp(
                    0, 2 * self.a_quantizer.n_levels - 1
                )
                dequant = (quant - cur_zero) * cur_scale
                error = (x_expanded - dequant) ** 2
                if x.ndim > 2:
                    error = error.mean(dim=list(range(1, error.ndim - 2)))
                if not self.a_quantizer.channel_wise:
                    error = error.mean(dim=-2, keepdim=True)
                error = error.sum(dim=0)
                scores[:, start:stop] -= error
        best = scores.argmax(dim=-1, keepdim=True)
        scale = torch.gather(scales, -1, best).squeeze(-1)
        zero = torch.gather(zero_points, -1, best).squeeze(-1)
        _assign_parameter(self.a_quantizer, "scale", scale.reshape(1))
        _assign_parameter(self.a_quantizer, "zero_point", zero.reshape(1))
        self.a_quantizer.inited = True

    def _search_weight_for_output(self, candidates, records, candidate_chunk):
        scales, zero_points = candidates
        scores = torch.zeros(
            scales.shape[0], self.n_groups, self.out_per_group, device=self.device, dtype=self.weight.dtype
        )
        for inputs, raw_output in iter_records(records, 1):
            x, raw = move_record(inputs, raw_output, self.device, self.weight.dtype)
            x_quant = self.quant_input(x[0])
            raw_grouped = raw.reshape(*raw.shape[:-1], self.n_groups, self.out_per_group)
            for start in range(0, scales.shape[0], candidate_chunk):
                stop = min(scales.shape[0], start + candidate_chunk)
                cur_scale = scales[start:stop]
                cur_zero = zero_points[start:stop]
                weight = self.weight_view().detach().unsqueeze(0)
                quant = (torch.round(weight / cur_scale) + cur_zero).clamp(
                    0, 2 * self.w_quantizer.n_levels - 1
                )
                dequant = ((quant - cur_zero) * cur_scale).reshape(-1, self.in_features)
                if self.bias is None:
                    bias = None
                else:
                    bias = self.bias.detach().repeat(stop - start)
                output = F.linear(x_quant, dequant, bias)
                output = output.reshape(*output.shape[:-1], stop - start, self.n_groups, self.out_per_group)
                target = raw_grouped.unsqueeze(-3)
                error = (target - output) ** 2
                if error.ndim > 4:
                    error = error.mean(dim=list(range(1, error.ndim - 3)))
                scores[start:stop] -= error.sum(dim=0)
        best = scores.argmax(dim=0, keepdim=True).unsqueeze(-1)
        best_scale = torch.gather(scales, 0, best).squeeze(0)
        best_zero = torch.gather(zero_points, 0, best).squeeze(0)
        _assign_parameter(self.w_quantizer, "scale", best_scale)
        _assign_parameter(self.w_quantizer, "zero_point", best_zero)

    def _search_activation_for_output(self, candidates, records, candidate_chunk):
        scales, zero_points = candidates
        n_candidates = eq_candidate_count(candidates)
        scores = torch.zeros(n_candidates, device=self.device, dtype=self.weight.dtype)
        quant_weight = self.quant_weight()
        for inputs, raw_output in iter_records(records, 1):
            x, raw = move_record(inputs, raw_output, self.device, self.weight.dtype)
            x = x[0]
            for start in range(0, n_candidates, candidate_chunk):
                stop = min(n_candidates, start + candidate_chunk)
                cur_scale = scales[:, start:stop]
                cur_zero = zero_points[:, start:stop]
                x_expanded = x.unsqueeze(-1)
                quant = (torch.round(x_expanded / cur_scale) + cur_zero).clamp(
                    0, 2 * self.a_quantizer.n_levels - 1
                )
                dequant = (quant - cur_zero) * cur_scale
                candidate_input = dequant.transpose(-1, -2)
                output = F.linear(candidate_input, quant_weight, self.bias)
                error = (raw.unsqueeze(-2) - output) ** 2
                error = error.mean(dim=-1)
                if error.ndim > 2:
                    error = error.mean(dim=list(range(1, error.ndim - 1)))
                scores[start:stop] -= error.sum(dim=0)
        best = int(scores.argmax().item())
        _assign_parameter(self.a_quantizer, "scale", scales[:, best].reshape(1))
        _assign_parameter(self.a_quantizer, "zero_point", zero_points[:, best].reshape(1))

    @torch.no_grad()
    def _affine_reparameterize(self, records, prev_layer, eq_n, candidate_chunk):
        if self.a_bit == 32:
            raise ValueError("prev_layer reparameterization requires a finite activation bit width")
        if not hasattr(prev_layer, "weight") or not hasattr(prev_layer, "bias"):
            raise TypeError("prev_layer must expose affine weight and bias parameters")
        if prev_layer.weight is None or prev_layer.bias is None:
            raise ValueError("prev_layer reparameterization requires affine weight and bias")

        channel_candidates = self._activation_candidates(records, eq_n, grouping="last_channel")
        self._initialize_activation_channelwise(channel_candidates, records, candidate_chunk)
        channel_scale = self.a_quantizer.scale.detach().reshape(-1)
        channel_zero = self.a_quantizer.zero_point.detach().reshape(-1)
        target_scale = channel_scale.mean().reshape(1)
        target_zero = channel_zero.mean().round().reshape(1)
        channel_min = -channel_zero * channel_scale
        target_min = -target_zero * target_scale
        ratio = channel_scale / target_scale
        offset = channel_min / ratio - target_min

        ratio_layer = ratio.to(device=prev_layer.weight.device, dtype=prev_layer.weight.dtype)
        offset_layer = offset.to(device=prev_layer.weight.device, dtype=prev_layer.weight.dtype)
        ratio_weight = ratio.to(device=self.weight.device, dtype=self.weight.dtype)
        offset_weight = offset.to(device=self.weight.device, dtype=self.weight.dtype)
        with torch.no_grad():
            prev_layer.weight.div_(ratio_layer)
            prev_layer.bias.div_(ratio_layer).sub_(offset_layer)
            self.weight.mul_(ratio_weight.reshape(1, -1))
            if self.bias is None:
                self.bias = nn.Parameter(torch.zeros(self.out_features, device=self.device, dtype=self.weight.dtype))
            self.bias.add_(torch.mv(self.weight, offset_weight))

        self.a_quantizer.channel_wise = False
        _assign_parameter(self.a_quantizer, "scale", target_scale.to(device=self.device, dtype=self.weight.dtype))
        _assign_parameter(self.a_quantizer, "zero_point", target_zero.to(device=self.device, dtype=self.weight.dtype))
        self.a_quantizer.inited = True
        return _TransformedRecords(records, ratio, offset)

    def _initialize_activation_channelwise(self, candidates, records, candidate_chunk):
        scales, zero_points = candidates
        n_candidates = eq_candidate_count(candidates)
        n_channels = scales.shape[0]
        scores = torch.zeros(n_channels, n_candidates, device=self.device, dtype=self.weight.dtype)
        for inputs, _raw_output in iter_records(records, 1):
            x = inputs[0].to(device=self.device, dtype=self.weight.dtype)
            x_expanded = x.unsqueeze(-1)
            for start in range(0, n_candidates, candidate_chunk):
                stop = min(n_candidates, start + candidate_chunk)
                cur_scale = scales[:, start:stop]
                cur_zero = zero_points[:, start:stop]
                quant = (torch.round(x_expanded / cur_scale) + cur_zero).clamp(
                    0, 2 * self.a_quantizer.n_levels - 1
                )
                dequant = (quant - cur_zero) * cur_scale
                error = (x_expanded - dequant) ** 2
                if x.ndim > 2:
                    error = error.mean(dim=list(range(1, error.ndim - 2)))
                scores[:, start:stop] -= error.sum(dim=0)
        best = scores.argmax(dim=-1, keepdim=True)
        _assign_parameter(self.a_quantizer, "scale", torch.gather(scales, -1, best).squeeze(-1))
        _assign_parameter(self.a_quantizer, "zero_point", torch.gather(zero_points, -1, best).squeeze(-1))
        self.a_quantizer.channel_wise = True
        self.a_quantizer.inited = True


def eq_candidate_count(candidates):
    return candidates[0].shape[-1] - 1
