# Copyright 2025 FIMA-Q authors.
#
# Calibration formulas are adapted from FIMA-Q (Apache-2.0); see
# docs/FIMA_Q_LICENSE.txt. This module keeps calibration records on the CPU and
# limits device transfers to a record and candidate chunk at a time.

"""Shared bounded-memory helpers for operator calibration."""

import os
import tempfile

import numpy as np
import torch


_EXACT_QUANTILE_LIMIT = 4_000_000


def validate_records(records, input_count):
    """Validate the replayable ``(inputs, raw_output)`` calibration protocol."""
    if iter(records) is records:
        raise TypeError("calibrate() requires a reusable iterable of records")
    iterator = iter(records)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("calibration records must contain at least one microbatch")
    if not isinstance(first, (tuple, list)) or len(first) != 2:
        raise TypeError("each record must be (tuple_of_cpu_inputs, cpu_raw_output)")
    inputs, raw_output = first
    if not isinstance(inputs, (tuple, list)) or len(inputs) != input_count:
        raise TypeError("record inputs must be a tuple with {} tensor(s)".format(input_count))
    tensors = list(inputs) + [raw_output]
    if any(not torch.is_tensor(t) for t in tensors):
        raise TypeError("calibration inputs and outputs must be tensors")
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("calibration records must be stored on CPU")
    return first


def iter_records(records, input_count):
    """Yield validated CPU record tensors without retaining a second cache."""
    for item in records:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("each record must be (tuple_of_cpu_inputs, cpu_raw_output)")
        inputs, raw_output = item
        if not isinstance(inputs, (tuple, list)) or len(inputs) != input_count:
            raise TypeError("record inputs must be a tuple with {} tensor(s)".format(input_count))
        tensors = list(inputs) + [raw_output]
        if any(not torch.is_tensor(t) for t in tensors):
            raise TypeError("calibration inputs and outputs must be tensors")
        if any(t.device.type != "cpu" for t in tensors):
            raise ValueError("calibration records must be stored on CPU")
        yield tuple(t.detach() for t in inputs), raw_output.detach()


def quantile_summary(records, input_count, input_index, probabilities, grouping="tensor"):
    """Compute exact pooled quantiles, spilling large tensors to a memmap.

    Small arrays use ``torch.quantile`` directly to retain its interpolation
    and dtype behavior. Large arrays are written to a disk-backed ``[groups,
    positions]`` array and selected exactly one group at a time with NumPy's
    in-place partition, then linearly interpolated like ``torch.quantile``.
    """
    total_positions = 0
    groups = None
    source_dtype = torch.float32

    def group_count_and_positions(x):
        if grouping == "tensor":
            return 1, x.numel()
        if grouping == "last_channel":
            if x.ndim < 1:
                raise ValueError("last_channel grouping needs a non-scalar tensor")
            return x.shape[-1], x.numel() // x.shape[-1]
        if grouping == "head":
            if x.ndim < 2:
                raise ValueError("head grouping needs a head axis at dimension 1")
            return x.shape[1], x.numel() // x.shape[1]
        raise ValueError("unknown quantile grouping: {}".format(grouping))

    for inputs, _raw_output in iter_records(records, input_count):
        x = inputs[input_index]
        record_groups, positions = group_count_and_positions(x)
        if groups is None:
            groups = record_groups
            if x.dtype == torch.float64:
                source_dtype = torch.float64
        elif groups != record_groups:
            raise ValueError("calibration records have inconsistent quantile group counts")
        total_positions += positions

    if groups is None or total_positions == 0:
        raise ValueError("cannot compute quantiles over empty calibration data")
    probability = torch.as_tensor(probabilities, dtype=source_dtype, device="cpu")

    if total_positions * groups <= _EXACT_QUANTILE_LIMIT:
        pieces = []
        for inputs, _raw_output in iter_records(records, input_count):
            x = inputs[input_index].to(dtype=source_dtype, device="cpu")
            if grouping == "tensor":
                matrix = x.reshape(1, -1)
            elif grouping == "last_channel":
                matrix = x.reshape(-1, x.shape[-1]).transpose(0, 1)
            else:
                matrix = x.transpose(0, 1).contiguous().reshape(x.shape[1], -1)
            pieces.append(matrix)
        pooled = torch.cat(pieces, dim=1)
        return torch.quantile(pooled, probability, dim=1)

    mapped_dtype = np.float64 if source_dtype == torch.float64 else np.float32
    parent = getattr(records, "root", None)
    if parent is not None:
        os.makedirs(parent, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="video-ptq-quantiles-", suffix=".mmap", dir=parent)
    os.close(fd)
    try:
        pooled = np.memmap(path, dtype=mapped_dtype, mode="w+", shape=(groups, total_positions))
        cursor = 0
        for inputs, _raw_output in iter_records(records, input_count):
            x = inputs[input_index].to(dtype=source_dtype, device="cpu")
            if grouping == "tensor":
                matrix = x.reshape(1, -1)
                count = matrix.shape[1]
                pooled[:, cursor:cursor + count] = matrix.numpy()
            elif grouping == "last_channel":
                matrix = x.reshape(-1, x.shape[-1])
                count = matrix.shape[0]
                pooled[:, cursor:cursor + count] = matrix.numpy().T
            else:
                count = x.numel() // groups
                for head in range(groups):
                    pooled[head, cursor:cursor + count] = x[:, head].reshape(-1).numpy()
            cursor += count
        pooled.flush()

        ranks = probability.numpy() * (total_positions - 1)
        lower = np.floor(ranks).astype(np.int64)
        upper = np.ceil(ranks).astype(np.int64)
        partition_indices = np.unique(np.concatenate([lower, upper]))
        result = np.empty((probability.numel(), groups), dtype=mapped_dtype)
        fraction = ranks - lower
        for group in range(groups):
            row = pooled[group]
            row.partition(partition_indices)
            low_value = np.asarray(row[lower])
            high_value = np.asarray(row[upper])
            result[:, group] = low_value + (high_value - low_value) * fraction
        pooled.flush()
        return torch.from_numpy(result.copy()).to(dtype=source_dtype)
    finally:
        try:
            del pooled
        except UnboundLocalError:
            pass
        try:
            os.remove(path)
        except OSError:
            pass


def move_record(inputs, raw_output, device, dtype=None):
    """Move one calibration record to the operator's own device."""
    if dtype is None:
        moved_inputs = tuple(x.to(device=device, non_blocking=False) for x in inputs)
        moved_output = raw_output.to(device=device, non_blocking=False)
    else:
        moved_inputs = tuple(x.to(device=device, dtype=dtype, non_blocking=False) for x in inputs)
        moved_output = raw_output.to(device=device, dtype=dtype, non_blocking=False)
    return moved_inputs, moved_output


def check_search_options(eq_n, search_round, candidate_chunk):
    if int(eq_n) <= 0:
        raise ValueError("eq_n must be positive")
    if int(search_round) < 0:
        raise ValueError("search_round must be nonnegative")
    if int(candidate_chunk) <= 0:
        raise ValueError("candidate_chunk must be positive")

