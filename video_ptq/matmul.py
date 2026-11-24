# Copyright 2025 FIMA-Q authors.
#
# Adapted from FIMA-Q's quant_layers/matmul.py (Apache-2.0). See
# docs/FIMA_Q_LICENSE.txt. Candidate statistics and output-error searches preserve
# its per-head asymmetric affine grid while streaming records and query rows.

"""Asymmetric fake-quant matrix multiplication for VideoMAE attention."""

import torch
from torch import nn

from ._calibration import check_search_options, iter_records, quantile_summary, validate_records
from .quantizers import UniformQuantizer


def _assign_parameter(module, name, value):
    setattr(module, name, nn.Parameter(value.detach().clone()))


class QuantMatMul(nn.Module):
    """Fake-quantize both operands of batched attention matrix multiplication."""

    def __init__(self, num_heads, a_bit=4, token_channel_wise=True, head_channel_wise=True):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        self.num_heads = int(num_heads)
        self.a_bit = int(a_bit)
        self.token_channel_wise = bool(token_channel_wise)
        self.head_channel_wise = bool(head_channel_wise)
        self.mode = "raw"
        self.calibrated = False
        groups = self.num_heads if self.head_channel_wise else 1
        shape = (1, groups, 1, 1)
        self.A_quantizer = UniformQuantizer(a_bit, symmetric=False, channel_wise=self.head_channel_wise)
        self.B_quantizer = UniformQuantizer(a_bit, symmetric=False, channel_wise=self.head_channel_wise)
        for quantizer in (self.A_quantizer, self.B_quantizer):
            _assign_parameter(quantizer, "scale", torch.zeros(shape))
            _assign_parameter(quantizer, "zero_point", torch.zeros(shape))

    @property
    def device(self):
        return self.A_quantizer.scale.device

    def forward(self, A, B):
        if self.mode == "raw":
            return A @ B
        if self.mode == "quant_forward":
            if not self.calibrated:
                raise RuntimeError("MatMul must be calibrated before quant_forward")
            return self.A_quantizer(A) @ self.B_quantizer(B)
        raise NotImplementedError("unsupported QuantMatMul mode: {}".format(self.mode))

    def quant_input_A(self, x):
        return self.A_quantizer(x)

    def quant_input_B(self, x):
        return self.B_quantizer(x)

    @torch.no_grad()
    def calibrate(self, records, *, eq_n, search_round, candidate_chunk, prev_layer=None):
        if prev_layer is not None:
            raise ValueError("QuantMatMul does not support prev_layer reparameterization")
        check_search_options(eq_n, search_round, candidate_chunk)
        validate_records(records, 2)
        if self.a_bit != 32:
            n_zero_points = min(16, 2 ** self.a_bit)
            if eq_n < n_zero_points or eq_n % n_zero_points:
                raise ValueError(
                    "eq_n must be a positive multiple of {} for the FIMA-Q zero-point grid".format(
                        n_zero_points
                    )
                )
        first_inputs, first_output = validate_records(records, 2)
        self._validate_shapes(first_inputs, first_output)
        if self.a_bit == 32:
            self.A_quantizer.inited = True
            self.B_quantizer.inited = True
            self.calibrated = True
            return self

        A_ranges = self._percentile_ranges(records, 0, eq_n)
        B_ranges = self._percentile_ranges(records, 1, eq_n)
        self._set_from_candidate(A_ranges, eq_n - 1, self.A_quantizer)
        self._set_from_candidate(B_ranges, eq_n - 1, self.B_quantizer)

        A_best = self._search_side("A", A_ranges, records, eq_n, candidate_chunk)
        B_best = self._search_side("B", B_ranges, records, eq_n, candidate_chunk)
        for _ in range(int(search_round)):
            for side, ranges, quantizer, best in (
                ("A", A_ranges, self.A_quantizer, A_best),
                ("B", B_ranges, self.B_quantizer, B_best),
            ):
                upper, lower = ranges
                for coordinate in range(2):
                    if coordinate == 0:
                        upper_current = self._gather_candidate(upper, best)
                        lower_current = lower
                    else:
                        upper_current = upper
                        lower_current = self._gather_candidate(lower, best)
                    best = self._search_side(
                        side, (upper_current, lower_current), records, eq_n, candidate_chunk
                    )
                if side == "A":
                    A_best = best
                else:
                    B_best = best

        if self.token_channel_wise:
            A, B = first_inputs
            if A.ndim == 4 and B.ndim == 4:
                if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1] or A.shape[-1] != B.shape[-2]:
                    raise ValueError("attention MatMul operands have incompatible batch/head/reduction shapes")
                groups = self.num_heads if self.head_channel_wise else 1
                if self.head_channel_wise:
                    a_scale = self.A_quantizer.scale.expand(1, groups, A.shape[-2], 1).clone()
                    b_scale = self.B_quantizer.scale.expand(1, groups, 1, B.shape[-1]).clone()
                else:
                    a_scale = self.A_quantizer.scale.expand(1, 1, A.shape[-2], 1).clone()
                    b_scale = self.B_quantizer.scale.expand(1, 1, 1, B.shape[-1]).clone()
                _assign_parameter(self.A_quantizer, "scale", a_scale)
                _assign_parameter(self.B_quantizer, "scale", b_scale)

        self.A_quantizer.inited = True
        self.B_quantizer.inited = True
        self.calibrated = True
        return self

    def _validate_shapes(self, inputs, output):
        A, B = inputs
        if A.ndim != 4 or B.ndim != 4 or output.ndim != 4:
            raise ValueError("QuantMatMul calibration expects [B,H,M,K] and [B,H,K,N] tensors")
        if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1] or A.shape[-1] != B.shape[-2]:
            raise ValueError("MatMul operands have incompatible batch/head/reduction shapes")
        if output.shape != (A.shape[0], A.shape[1], A.shape[-2], B.shape[-1]):
            raise ValueError("raw output shape does not match A @ B")
        if self.head_channel_wise and A.shape[1] != self.num_heads:
            raise ValueError("input head count does not match num_heads")

    def _percentile_ranges(self, records, input_index, eq_n):
        grouping = "head" if self.head_channel_wise else "tensor"
        quantiles = quantile_summary(
            records,
            2,
            input_index,
            [0.99, 0.99999, 0.01, 0.00001],
            grouping=grouping,
        ).to(device=self.device, dtype=self.A_quantizer.scale.dtype)
        upper_low, upper_high, lower_high, lower_low = quantiles
        n_groups = quantiles.shape[1]
        grid = torch.linspace(0.0, 1.0, steps=int(eq_n) + 1, device=self.device, dtype=quantiles.dtype)
        upper = upper_low.unsqueeze(0) + grid[:, None] * (upper_high - upper_low).unsqueeze(0)
        lower = lower_low.unsqueeze(0) + grid[:, None] * (lower_high - lower_low).unsqueeze(0)
        upper = upper.reshape(eq_n + 1, 1, n_groups, 1, 1)
        lower = lower.reshape(eq_n + 1, 1, n_groups, 1, 1)
        return upper, lower

    def _make_candidates(self, ranges):
        upper, lower = ranges
        denominator = float(2 * self.A_quantizer.n_levels - 1)
        scale = ((upper - lower) / denominator).clamp_min(1e-8)
        zero = torch.round(-lower / scale)
        return scale, zero

    def _set_from_candidate(self, ranges, index, quantizer):
        scale_candidates, zero_candidates = self._make_candidates(ranges)
        if torch.is_tensor(index):
            scale = self._gather_candidate(scale_candidates, index).squeeze(0)
            zero = self._gather_candidate(zero_candidates, index).squeeze(0)
        else:
            scale = scale_candidates[index]
            zero = zero_candidates[index]
        scale = scale.reshape_as(quantizer.scale)
        zero = zero.reshape_as(quantizer.zero_point)
        _assign_parameter(quantizer, "scale", scale)
        _assign_parameter(quantizer, "zero_point", zero)
        quantizer.inited = True

    def _gather_candidate(self, values, indices):
        index = indices.reshape(1, 1, -1, 1, 1)
        return torch.gather(values, dim=0, index=index)

    def _search_side(self, side, ranges, records, eq_n, candidate_chunk):
        scale_candidates, zero_candidates = self._make_candidates(ranges)
        group_count = scale_candidates.shape[2]
        scores = torch.zeros(int(eq_n), group_count, device=self.device, dtype=scale_candidates.dtype)
        for inputs, raw_output in iter_records(records, 2):
            A_cpu, B_cpu = inputs
            self._validate_shapes((A_cpu, B_cpu), raw_output)
            A, B = A_cpu, B_cpu
            query_count = A.shape[-2]
            batch, heads = A.shape[:2]
            reduction_width = A.shape[-1]
            output_width = B.shape[-1]
            bytes_per_element = max(A.element_size(), B.element_size(), raw_output.element_size())
            record_bytes = (
                A.numel() * A.element_size()
                + B.numel() * B.element_size()
                + raw_output.numel() * raw_output.element_size()
            )
            if record_bytes <= 256 * 1024 * 1024:
                A_source = A.to(device=self.device)
                raw_source = raw_output.to(device=self.device)
            else:
                A_source = A
                raw_source = raw_output
            candidate_tensor_budget = 64 * 1024 * 1024
            bytes_per_row = max(1, batch * heads * max(reduction_width, output_width) * bytes_per_element)
            if side == "A":
                B_device = B.to(device=self.device)
                fixed_B = self.B_quantizer(B_device)
            else:
                fixed_B = None
            chunk_limit = min(int(candidate_chunk), int(eq_n))
            fixed_candidate_bytes = 0
            if side == "B":
                # B candidate Q/DQ occupies a complete right-hand operand.
                bytes_per_full_candidate = max(1, batch * heads * reduction_width * output_width * bytes_per_element * 2)
                chunk_limit = min(chunk_limit, max(1, candidate_tensor_budget // bytes_per_full_candidate))
                fixed_candidate_bytes = chunk_limit * bytes_per_full_candidate
                B_device = B.to(device=self.device)
            remaining_budget = max(bytes_per_row, candidate_tensor_budget - fixed_candidate_bytes)
            query_chunk = max(1, remaining_budget // max(1, chunk_limit * bytes_per_row))
            query_chunk = min(query_count, int(query_chunk))
            for row_start in range(0, query_count, query_chunk):
                row_stop = min(query_count, row_start + query_chunk)
                target = raw_source[:, :, row_start:row_stop, :]
                row_A = A_source[:, :, row_start:row_stop, :]
                if row_A.device != self.device:
                    row_A = row_A.to(device=self.device)
                if target.device != self.device:
                    target = target.to(device=self.device)
                target = target.unsqueeze(0)
                if side == "B":
                    fixed_row_A = self.A_quantizer(row_A).unsqueeze(0)
                for start in range(0, int(eq_n), chunk_limit):
                    stop = min(int(eq_n), start + chunk_limit)
                    cur_scale = scale_candidates[start:stop]
                    cur_zero = zero_candidates[start:stop]
                    if side == "A":
                        cur_A = self._quantize_candidates(row_A, cur_scale, cur_zero)
                        result = cur_A @ fixed_B.unsqueeze(0)
                    else:
                        cur_B = self._quantize_candidates(B_device, cur_scale, cur_zero)
                        result = fixed_row_A @ cur_B
                    error = (target - result) ** 2
                    if self.head_channel_wise:
                        error = error.mean(dim=tuple(range(3, error.ndim)))
                    else:
                        error = error.mean(dim=tuple(range(2, error.ndim)))
                    scores[start:stop] -= error.sum(dim=1) * ((row_stop - row_start) / float(query_count))
        best = scores.argmax(dim=0)
        self._set_from_candidate(ranges, best, self.A_quantizer if side == "A" else self.B_quantizer)
        return best

    def _quantize_candidates(self, x, scale, zero_point):
        quant = (torch.round(x.unsqueeze(0) / scale) + zero_point).clamp(
            0, 2 * self.A_quantizer.n_levels - 1
        )
        return (quant - zero_point) * scale
