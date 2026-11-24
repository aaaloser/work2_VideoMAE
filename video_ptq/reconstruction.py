# Adapted from FIMA-Q's utils/block_recon.py (Apache-2.0).
# See docs/FIMA_Q_LICENSE.txt for the license.

"""FIMA-Q block reconstruction for explicitly registered modules.

Loader/cache tensors stay on CPU (or in a temporary disk-backed cache when
cache_dir is supplied); only bounded forward microbatches move to the device.
batch_size is the logical optimizer batch; microbatch_size bounds its forwards.
"""

import math
import logging
import os
import tempfile

import torch
import torch.nn.functional as F


def teacher_probabilities(logits, temperature=20.0):
    """Return the raw model's temperature-scaled teacher distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.softmax(logits.float() / float(temperature), dim=-1).detach()


def fisher_kl_loss(student_logits, teacher_probs, temperature=20.0,
                   logical_batch_size=1):
    """KL contribution with the denominator of its logical loader batch.

    logical_batch_size may be a scalar or one denominator per sample. The
    vector form lets physical forward chunks reproduce a larger batchmean KL,
    including a shorter final logical batch.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_probs = F.log_softmax(student_logits.float() / float(temperature), dim=-1)
    per_sample = F.kl_div(
        log_probs,
        teacher_probs.to(device=log_probs.device, dtype=log_probs.dtype),
        reduction="none",
    ).reshape(student_logits.shape[0], -1).sum(dim=1)
    if torch.is_tensor(logical_batch_size) or isinstance(logical_batch_size, (list, tuple)):
        denominators = torch.as_tensor(logical_batch_size).to(
            device=per_sample.device, dtype=per_sample.dtype
        ).reshape(-1)
        if denominators.numel() != per_sample.numel() or bool((denominators <= 0).any()):
            raise ValueError("logical batch denominators must be positive and match samples")
        return (per_sample / denominators).sum()
    if logical_batch_size <= 0:
        raise ValueError("logical_batch_size must be positive")
    return per_sample.sum() / float(logical_batch_size)


def head_kl(student_logits, teacher_logits):
    """FIMA-Q head reconstruction KL on original, unscaled logits."""
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_probs = F.softmax(teacher_logits.float(), dim=-1).detach()
    per_sample = F.kl_div(student_log_probs, teacher_probs, reduction="none")
    return per_sample.reshape(student_logits.shape[0], -1).sum(dim=1).mean()


def dplr_terms(error, raw_grad, inverse_B):
    """Compute FIMA-Q low-rank and diagonal DPLR terms without a d-by-d matrix."""
    if error.ndim < 2:
        raise ValueError("error must include a batch and feature dimension")
    error_flat = error.abs().reshape(error.shape[0], -1).float()
    grad_flat = raw_grad.abs().reshape(raw_grad.shape[0], -1).to(
        device=error_flat.device, dtype=error_flat.dtype
    )
    inverse = inverse_B.to(device=error_flat.device, dtype=error_flat.dtype)
    if error_flat.shape[1] != grad_flat.shape[1]:
        raise ValueError("output error and Fisher gradients have different feature counts")
    if tuple(inverse.shape) != (grad_flat.shape[0], grad_flat.shape[0]):
        raise ValueError("inverse_B shape does not match the Fisher rank")
    projections = error_flat.matmul(grad_flat.transpose(0, 1))
    low_rank = (projections.matmul(inverse) * projections).sum(dim=1).mean()
    diagonal = (error_flat.square() * grad_flat.mean(dim=0)).mean()
    return low_rank, diagonal


def normalized_dplr_loss(error, raw_grad, inverse_B, reference, p1=1.0, p2=1.0):
    """Apply FIMA-Q's independently normalized and weighted DPLR terms."""
    low_rank, diagonal = dplr_terms(error, raw_grad, inverse_B)
    ref_low = torch.as_tensor(reference[0], device=low_rank.device, dtype=low_rank.dtype)
    ref_diag = torch.as_tensor(reference[1], device=diagonal.device, dtype=diagonal.dtype)
    if bool(ref_low.detach().abs() == 0) or bool(ref_diag.detach().abs() == 0):
        raise ValueError("DPLR reference term is zero; cannot normalize reconstruction loss")
    low_norm = low_rank / ref_low
    diag_norm = diagonal / ref_diag
    return float(p1) * low_norm + float(p2) * diag_norm


def _freeze_id(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, list):
        return tuple(_freeze_id(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_id(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _normalise_ids(raw_ids, count, start):
    if raw_ids is None:
        return [("synthetic-clip", start + index) for index in range(count)]
    if torch.is_tensor(raw_ids):
        raw_ids = raw_ids.detach().cpu().tolist()
    elif isinstance(raw_ids, (str, bytes)):
        raw_ids = [raw_ids] if count == 1 else None
    elif not isinstance(raw_ids, (list, tuple)):
        raw_ids = [raw_ids] if count == 1 else None
    if raw_ids is None or len(raw_ids) != count:
        raise ValueError("loader IDs must contain one stable ID per clip")
    return [_freeze_id(value) for value in raw_ids]


def _unpack_batch(batch, start):
    if torch.is_tensor(batch):
        clips, raw_ids = batch, None
    elif (isinstance(batch, (tuple, list)) and len(batch) == 2
          and torch.is_tensor(batch[0])):
        clips, raw_ids = batch
    else:
        raise TypeError("loader batches must be clips or (clips, stable IDs)")
    if clips.ndim < 1:
        raise ValueError("clips must have a batch dimension")
    ids = _normalise_ids(raw_ids, int(clips.shape[0]), start)
    return clips, ids


def _loader_chunks(loader, microbatch_size):
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    if torch.is_tensor(loader):
        source = (loader,)
    else:
        try:
            iterator = iter(loader)
        except TypeError:
            raise TypeError("loader must be a reiterable iterable")
        if iterator is loader:
            raise ValueError("loader must be reiterable")
        source = iterator
    offset = 0
    for batch in source:
        clips, ids = _unpack_batch(batch, offset)
        batch_count = int(clips.shape[0])
        for begin in range(0, batch_count, microbatch_size):
            end = min(begin + microbatch_size, batch_count)
            yield clips[begin:end], ids[begin:end]
        offset += batch_count


def _assert_unique_ids(ids, label):
    if len(ids) != len(set(ids)):
        raise ValueError("stable clip IDs must be unique within a loader pass ({})".format(label))


def _assert_same_id_set(ids, expected, label):
    _assert_unique_ids(ids, label)
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError("loader IDs changed between reconstruction passes ({})".format(label))


class _SampleStore(object):
    """Append-only CPU tensor storage with optional disk-backed mapping."""

    def __init__(self, path=None):
        self.path = path
        self._chunks = []
        self._tensor = None
        self._mapped = None
        self._mapped_path = None
        self._generation = 0
        self._capacity = 0
        self._count = 0
        self._shape = None
        self._dtype = None
        self._sample_numel = 0

    def _allocate(self, capacity):
        old_map = self._mapped
        old_path = self._mapped_path
        path = "{}.{}.bin".format(self.path, self._generation)
        self._generation += 1
        element_size = torch.empty((), dtype=self._dtype).element_size()
        with open(path, "wb") as mapped_file:
            mapped_file.truncate(capacity * self._sample_numel * element_size)
        mapped = torch.from_file(
            path, shared=True, size=capacity * self._sample_numel, dtype=self._dtype
        ).view((capacity,) + tuple(self._shape))
        if old_map is not None and self._count:
            mapped[:self._count].copy_(old_map[:self._count])
        self._mapped = mapped
        self._mapped_path = path
        self._capacity = capacity
        del old_map
        if old_path is not None:
            try:
                os.remove(old_path)
            except OSError:
                pass

    def append(self, values):
        values = values.detach().to(device="cpu").contiguous()
        if values.shape[0] == 0:
            return
        if self._shape is None:
            self._shape = tuple(values.shape[1:])
            self._dtype = values.dtype
            self._sample_numel = int(values[0].numel())
        elif tuple(values.shape[1:]) != self._shape or values.dtype != self._dtype:
            raise ValueError("cached tensor shape or dtype changed during replay")
        if self.path is None:
            self._chunks.append(values)
        else:
            required = self._count + int(values.shape[0])
            if required > self._capacity:
                self._allocate(max(required, max(1, self._capacity * 2)))
            self._mapped[self._count:required].copy_(values)
        self._count += int(values.shape[0])

    def finalize(self):
        if self.path is None and self._tensor is None:
            if not self._chunks:
                raise ValueError("cannot finalize an empty tensor cache")
            self._tensor = torch.cat(self._chunks, dim=0)
            self._chunks = []
        if self._count == 0:
            raise ValueError("cannot finalize an empty tensor cache")

    def gather(self, rows):
        if not rows:
            raise ValueError("cannot gather an empty sample batch")
        self.finalize()
        indices = torch.as_tensor(rows, dtype=torch.long)
        source = self._mapped if self.path is not None else self._tensor
        return torch.index_select(source, 0, indices)

    def close(self):
        self._mapped = None
        self._tensor = None
        self._chunks = []
        if self._mapped_path is not None:
            try:
                os.remove(self._mapped_path)
            except OSError:
                pass
            self._mapped_path = None


class _AlignedStore(object):
    def __init__(self, store, ids):
        self.store = store
        self.row_by_id = {key: row for row, key in enumerate(ids)}

    def gather_positions(self, positions, canonical_ids):
        ids = [canonical_ids[int(position)] for position in positions]
        return self.store.gather([self.row_by_id[key] for key in ids])

    def gather_ids(self, ids):
        return self.store.gather([self.row_by_id[key] for key in ids])


def _tensor_store(path, stem):
    return _SampleStore(None if path is None else os.path.join(path, stem))


def _set_model_modes(model, units, quantized_indices=()):
    for module in model.modules():
        if hasattr(module, "mode"):
            module.mode = "raw"
    for index in quantized_indices:
        for module in units[index].module.modules():
            if hasattr(module, "mode"):
                module.mode = "quant_forward"


def _capture_unit_data(model, units, unit_index, loader, microbatch_size,
                       cache_path, expected_ids, device):
    unit = units[unit_index]
    raw_input_store = _tensor_store(cache_path, "unit_{:03d}_raw_input".format(unit_index))
    raw_output_store = _tensor_store(cache_path, "unit_{:03d}_raw_output".format(unit_index))
    pass_ids = []
    current_inputs, current_outputs = [], []

    def pre_hook(_module, args):
        if not args or not torch.is_tensor(args[0]):
            raise TypeError("registered reconstruction units must take one tensor input")
        current_inputs.append(args[0].detach())

    def out_hook(_module, _args, output):
        if not torch.is_tensor(output):
            raise TypeError("registered reconstruction units must return one tensor output")
        current_outputs.append(output.detach())

    _set_model_modes(model, units)
    pre_handle = unit.module.register_forward_pre_hook(pre_hook)
    out_handle = unit.module.register_forward_hook(out_hook)
    try:
        with torch.no_grad():
            for clips, ids in _loader_chunks(loader, microbatch_size):
                current_inputs[:] = []
                current_outputs[:] = []
                model(clips.to(device))
                if len(current_inputs) != 1 or len(current_outputs) != 1:
                    raise RuntimeError("unit {} must execute once per forward".format(unit.name))
                raw_input_store.append(current_inputs[0])
                raw_output_store.append(current_outputs[0])
                pass_ids.extend(ids)
    finally:
        pre_handle.remove()
        out_handle.remove()
    _assert_same_id_set(pass_ids, expected_ids, "raw cache for {}".format(unit.name))
    raw_input_store.finalize()
    raw_output_store.finalize()
    return _AlignedStore(raw_input_store, pass_ids), _AlignedStore(raw_output_store, pass_ids)


def _capture_quantized_input(model, units, unit_index, loader, microbatch_size,
                             cache_path, expected_ids, raw_input_store, device):
    if unit_index == 0:
        return raw_input_store
    unit = units[unit_index]
    store = _tensor_store(cache_path, "unit_{:03d}_quant_input".format(unit_index))
    pass_ids = []
    captured_inputs = []

    def pre_hook(_module, args):
        if not args or not torch.is_tensor(args[0]):
            raise TypeError("registered reconstruction units must take one tensor input")
        captured_inputs.append(args[0].detach())

    _set_model_modes(model, units, range(unit_index))
    handle = unit.module.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            for clips, ids in _loader_chunks(loader, microbatch_size):
                captured_inputs[:] = []
                model(clips.to(device))
                if len(captured_inputs) != 1:
                    raise RuntimeError("unit {} must execute once per forward".format(unit.name))
                store.append(captured_inputs[0])
                pass_ids.extend(ids)
    finally:
        handle.remove()
        _set_model_modes(model, units)
    _assert_same_id_set(pass_ids, expected_ids, "quantized input for {}".format(unit.name))
    store.finalize()
    return _AlignedStore(store, pass_ids)


def _teacher_cache(model, units, loader, microbatch_size, device, temperature):
    _set_model_modes(model, units)
    probs_by_id = {}
    ordered_ids = []
    with torch.no_grad():
        for clips, ids in _loader_chunks(loader, microbatch_size):
            logits = model(clips.to(device))
            if not torch.is_tensor(logits) or logits.shape[0] != len(ids):
                raise TypeError("model must return one logits tensor per clip")
            probs = teacher_probabilities(logits, temperature).cpu()
            for key, probabilities in zip(ids, probs):
                if key in probs_by_id:
                    raise ValueError("stable clip IDs must be unique within a loader pass")
                probs_by_id[key] = probabilities
                ordered_ids.append(key)
    if not ordered_ids:
        raise ValueError("reconstruction loader is empty")
    return ordered_ids, probs_by_id


def _quantizers_in(module):
    for child in module.modules():
        for name in ("a_quantizer", "A_quantizer", "B_quantizer"):
            quantizer = getattr(child, name, None)
            if quantizer is not None:
                yield quantizer


def _set_qdrop(unit, probability):
    for quantizer in _quantizers_in(unit.module):
        if hasattr(quantizer, "drop_prob"):
            quantizer.drop_prob = float(probability)


def _init_quantizer_training(unit):
    for child in unit.module.modules():
        for name in ("w_quantizer", "a_quantizer", "A_quantizer", "B_quantizer"):
            quantizer = getattr(child, name, None)
            if quantizer is not None and callable(getattr(quantizer, "init_training", None)):
                quantizer.init_training()


def _rounding_regularizer(unit, count, iterations, warmup=0.2,
                          weight=0.01, start_b=20.0, end_b=2.0):
    if count < iterations * warmup:
        return None
    start = iterations * warmup
    if iterations <= start:
        b = end_b
    else:
        fraction = min(max((count - start) / (iterations - start), 0.0), 1.0)
        b = end_b + (start_b - end_b) * (1.0 - fraction)
    loss = None
    for child in unit.module.modules():
        quantizer = getattr(child, "w_quantizer", None)
        if quantizer is None or not hasattr(quantizer, "get_soft_targets"):
            continue
        targets = quantizer.get_soft_targets()
        term = float(weight) * (1.0 - (2.0 * (targets - 0.5).abs()).pow(b)).sum()
        loss = term if loss is None else loss + term
    return loss


def _dplr_update_due(iteration, iterations, rank, dis_mode):
    if dis_mode == "q":
        interval = max(1, int(math.floor(float(iterations) / float(rank))))
        return iteration % interval == 0
    return iteration < rank


def _estimate_fisher(model, units, unit_index, loader, microbatch_size,
                     fisher_batch_size, expected_ids, id_to_position,
                     teacher_by_id, raw_output_store, temperature,
                     device, current_grad_rows, current_delta_rows):
    unit = units[unit_index]
    _set_model_modes(model, units, (unit_index,))
    grad_sum = None
    delta_sum = None
    seen_ids = []
    output_values = []
    handle = unit.module.register_forward_hook(
        lambda _module, _args, output: output_values.append(output)
    )
    try:
        for clips, ids in _loader_chunks(loader, microbatch_size):
            seen_ids.extend(ids)
            teacher = torch.stack([teacher_by_id[key] for key in ids]).to(device)
            positions = [id_to_position[key] for key in ids]
            group_sizes = [
                min(fisher_batch_size,
                    len(expected_ids) - (position // fisher_batch_size) * fisher_batch_size)
                for position in positions
            ]
            output_values[:] = []
            logits = model(clips.to(device))
            if len(output_values) != 1:
                raise RuntimeError("unit {} must execute once during Fisher replay".format(unit.name))
            block_output = output_values[0]
            if not torch.is_tensor(block_output) or not block_output.requires_grad:
                raise RuntimeError("unit {} output must participate in the Fisher gradient".format(unit.name))
            loss = fisher_kl_loss(logits, teacher, temperature, group_sizes)
            block_grad = torch.autograd.grad(loss, block_output, retain_graph=False)[0]
            grad_values = block_grad.detach().float().reshape(len(ids), -1).abs()
            raw_values = raw_output_store.gather_ids(ids).to(device).float().reshape(len(ids), -1)
            delta_values = (
                block_output.detach().float().reshape(len(ids), -1) - raw_values
            ).abs()
            chunk_grad_sum = grad_values.sum(dim=0)
            chunk_delta_sum = delta_values.sum(dim=0)
            grad_sum = chunk_grad_sum if grad_sum is None else grad_sum + chunk_grad_sum
            delta_sum = chunk_delta_sum if delta_sum is None else delta_sum + chunk_delta_sum
            del logits, loss, block_output, block_grad, grad_values, raw_values, delta_values
    finally:
        handle.remove()
        _set_model_modes(model, units)
    _assert_same_id_set(seen_ids, expected_ids, "Fisher replay for {}".format(unit.name))
    divisor = float(len(expected_ids))
    grad_row = (grad_sum / divisor).unsqueeze(0)
    delta_row = (delta_sum / divisor).unsqueeze(0)
    if current_grad_rows is not None:
        grad_rows = torch.cat((current_grad_rows, grad_row), dim=0)
        delta_rows = torch.cat((current_delta_rows, delta_row), dim=0)
    else:
        grad_rows, delta_rows = grad_row, delta_row
    inverse_B = torch.linalg.inv(delta_rows.matmul(delta_rows.transpose(0, 1)))
    return grad_rows, delta_rows, inverse_B


def _optimizer_batch(raw_inputs, raw_outputs, quant_inputs,
                     canonical_ids, batch_indices, mode, drop_prob, device):
    raw_input = raw_inputs.gather_positions(batch_indices, canonical_ids).to(device)
    raw_output = raw_outputs.gather_positions(batch_indices, canonical_ids).to(device)
    if mode == "rinp":
        current_input = raw_input
    else:
        quant_input = quant_inputs.gather_positions(batch_indices, canonical_ids).to(device)
        if mode == "qdrop":
            mask = torch.rand_like(quant_input) < float(drop_prob)
            current_input = torch.where(mask, quant_input, raw_input)
        else:
            current_input = quant_input
    return current_input, raw_output


def _reference_for_batch(unit, current_input, target_output, raw_grad,
                         inverse_B, microbatch_size):
    total = int(current_input.shape[0])
    low_sum = None
    diagonal_sum = None
    with torch.no_grad():
        for begin in range(0, total, microbatch_size):
            end = min(begin + microbatch_size, total)
            output = unit.module(current_input[begin:end])
            low, diagonal = dplr_terms(
                output - target_output[begin:end], raw_grad, inverse_B
            )
            count = end - begin
            low_sum = low * count if low_sum is None else low_sum + low * count
            diagonal_sum = (diagonal * count if diagonal_sum is None
                            else diagonal_sum + diagonal * count)
    return low_sum / float(total), diagonal_sum / float(total)


def _reconstruct_unit(model, units, unit_index, loader, expected_ids,
                      teacher_by_id, cache_path, device, iterations,
                      batch_size, microbatch_size, rank, temperature,
                      drop_prob, fisher_batch_size, mode, dis_mode, p1, p2):
    # AdaRound remains a reconstruction-only dependency.
    from .quantizers import AdaRoundQuantizer

    unit = units[unit_index]
    logging.info("Reconstructing unit %s (%d/%d)", unit.name, unit_index + 1, len(units))
    raw_inputs, raw_outputs = _capture_unit_data(
        model, units, unit_index, loader, microbatch_size,
        cache_path, expected_ids, device
    )
    quant_inputs = _capture_quantized_input(
        model, units, unit_index, loader, microbatch_size,
        cache_path, expected_ids, raw_inputs, device
    )
    wrapped = []
    weight_parameters = []
    activation_parameters = []
    activation_requires_grad = {}
    activation_backups = {}
    original_parameter_flags = {
        parameter: parameter.requires_grad for parameter in model.parameters()
    }
    completed = False
    try:
        for child in unit.module.modules():
            if not hasattr(child, "w_quantizer"):
                continue
            weight_view_fn = getattr(child, "weight_view", None)
            if not callable(weight_view_fn):
                raise TypeError("quantized weight operators must implement weight_view()")
            original_quantizer = child.w_quantizer
            adaptive_quantizer = AdaRoundQuantizer(
                uq=original_quantizer,
                weight_tensor=weight_view_fn(),
                round_mode="learned_hard_sigmoid",
            )
            adaptive_quantizer.scale.requires_grad_(False)
            if isinstance(getattr(adaptive_quantizer, "zero_point", None), torch.nn.Parameter):
                adaptive_quantizer.zero_point.requires_grad_(False)
            adaptive_quantizer.soft_targets = True
            child.w_quantizer = adaptive_quantizer
            wrapped.append((child, original_quantizer, adaptive_quantizer))
            weight_parameters.append(adaptive_quantizer.alpha)

        for quantizer in _quantizers_in(unit.module):
            scale = getattr(quantizer, "scale", None)
            if torch.is_tensor(scale) and isinstance(scale, torch.nn.Parameter):
                activation_requires_grad[scale] = scale.requires_grad
                activation_backups[scale] = scale.detach().clone()
                scale.requires_grad_(True)
                if all(scale is not old for old in activation_parameters):
                    activation_parameters.append(scale)
        if not weight_parameters and not activation_parameters:
            raise ValueError("reconstruction unit has no trainable quantization parameters")

        _init_quantizer_training(unit)
        _set_qdrop(unit, drop_prob if mode == "qdrop" else 1.0)
        _set_model_modes(model, units, (unit_index,))

        # Only AdaRound alpha and activation scales receive optimizer gradients.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in weight_parameters + activation_parameters:
            parameter.requires_grad_(True)
        weight_optimizer = (
            torch.optim.Adam(weight_parameters, lr=1e-3)
            if weight_parameters else None
        )
        activation_optimizer = (
            torch.optim.Adam(activation_parameters, lr=4e-5)
            if activation_parameters else None
        )
        activation_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                activation_optimizer, T_max=iterations, eta_min=0.0
            ) if activation_optimizer is not None else None
        )

        grad_rows = None
        delta_rows = None
        inverse_B = None
        reference = None
        id_to_position = {key: index for index, key in enumerate(expected_ids)}
        is_head = bool(unit.is_head)
        fisher_refresh_count = 0

        for iteration in range(iterations):
            current_indices = None
            if not is_head and _dplr_update_due(iteration, iterations, rank, dis_mode):
                fisher_refresh_count += 1
                logging.info(
                    "Fisher refresh %d for %s at iteration %d/%d",
                    fisher_refresh_count, unit.name, iteration + 1, iterations,
                )
                grad_rows, delta_rows, inverse_B = _estimate_fisher(
                    model, units, unit_index, loader, microbatch_size,
                    fisher_batch_size, expected_ids, id_to_position,
                    teacher_by_id, raw_outputs, temperature, device,
                    grad_rows, delta_rows,
                )
                # Fisher replay restores every operator to raw mode; the
                # following reference and optimization forwards use this unit
                # quantized while all other units remain raw.
                _set_model_modes(model, units, (unit_index,))
                draw = torch.randperm(len(expected_ids))[:min(batch_size, len(expected_ids))]
                current_indices = [int(index) for index in draw.tolist()]
                current_input, current_target = _optimizer_batch(
                    raw_inputs, raw_outputs, quant_inputs, expected_ids,
                    current_indices, mode, drop_prob, device
                )
                # Match FIMA-Q's reference refresh to the first optimization
                # batch after each Fisher update. Replaying identical RNG state
                # makes QDrop's activation masks match the gradient pass.
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                reference = _reference_for_batch(
                    unit, current_input, current_target, grad_rows,
                    inverse_B, microbatch_size
                )
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, device)

            if current_indices is None:
                draw = torch.randperm(len(expected_ids))[:min(batch_size, len(expected_ids))]
                current_indices = [int(index) for index in draw.tolist()]
                current_input, current_target = _optimizer_batch(
                    raw_inputs, raw_outputs, quant_inputs, expected_ids,
                    current_indices, mode, drop_prob, device
                )

            logical_count = len(current_indices)
            if weight_optimizer is not None:
                weight_optimizer.zero_grad()
            if activation_optimizer is not None:
                activation_optimizer.zero_grad()
            for begin in range(0, logical_count, microbatch_size):
                end = min(begin + microbatch_size, logical_count)
                prediction = unit.module(current_input[begin:end])
                count = end - begin
                if is_head:
                    chunk_loss = head_kl(prediction, current_target[begin:end])
                else:
                    if grad_rows is None or inverse_B is None or reference is None:
                        raise RuntimeError("Fisher/DPLR state was not initialized")
                    chunk_loss = normalized_dplr_loss(
                        prediction - current_target[begin:end],
                        grad_rows,
                        inverse_B,
                        reference,
                        p1=p1,
                        p2=p2,
                    )
                (chunk_loss * (float(count) / float(logical_count))).backward()

            rounding_loss = _rounding_regularizer(unit, iteration + 1, iterations)
            if rounding_loss is not None:
                rounding_loss.backward()
            if weight_optimizer is not None:
                weight_optimizer.step()
            if activation_optimizer is not None:
                activation_optimizer.step()
                activation_scheduler.step()

        completed = True
    finally:
        for child, original_quantizer, adaptive_quantizer in wrapped:
            adaptive_quantizer.soft_targets = False
            if not completed:
                child.w_quantizer = original_quantizer
        if not completed:
            with torch.no_grad():
                for scale, old_value in activation_backups.items():
                    scale.copy_(old_value)
        for scale, requires_grad in activation_requires_grad.items():
            scale.requires_grad_(requires_grad)
        for quantizer in _quantizers_in(unit.module):
            if callable(getattr(quantizer, "end_training", None)):
                quantizer.end_training()
            if hasattr(quantizer, "drop_prob"):
                quantizer.drop_prob = 1.0
        for parameter, requires_grad in original_parameter_flags.items():
            parameter.requires_grad_(requires_grad)
        _set_model_modes(model, units, (unit_index,) if completed else ())
        raw_inputs.store.close()
        raw_outputs.store.close()
        if quant_inputs.store is not raw_inputs.store:
            quant_inputs.store.close()
    if completed:
        logging.info("Finished reconstruction unit %s", unit.name)
    return wrapped


def _harden_adaround_quantizers(wrapped):
    """Bake all learned rounds only after every unit has used the FP prefix."""
    planned = []
    with torch.no_grad():
        for child, original_quantizer, adaptive_quantizer in wrapped:
            weight_view = child.weight_view()
            hard_value = adaptive_quantizer.get_hard_value(weight_view).reshape_as(weight_view)
            planned.append((
                child,
                original_quantizer,
                adaptive_quantizer,
                weight_view,
                hard_value.detach(),
                weight_view.detach().clone(),
            ))
        copied = []
        try:
            for item in planned:
                item[3].copy_(item[4])
                copied.append(item)
        except Exception:
            for item in copied:
                item[3].copy_(item[5])
            raise
        for child, original_quantizer, adaptive_quantizer, _view, _hard, _old in planned:
            adaptive_quantizer.soft_targets = False
            child.w_quantizer = original_quantizer


def reconstruct(model, units, loader, *, device, iterations=20000,
                batch_size=32, rank=5, temperature=20, drop_prob=0.5,
                cache_dir=None, fisher_batch_size=32, mode="qdrop",
                dis_mode="q", p1=1.0, p2=1.0, microbatch_size=None):
    """Reconstruct registered units in order and return the quantized model.

    Loader batches may be physically smaller than fisher_batch_size. Fisher KL
    uses per-ID logical group denominators across the full sample set, with the
    final short group using its actual size. Model forwards and optimizer
    updates are separately bounded by microbatch_size and batch_size.
    """
    if iterations <= 0 or batch_size <= 0 or rank <= 0 or fisher_batch_size <= 0:
        raise ValueError("iterations, batch_size, rank, and fisher_batch_size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError("drop_prob must be in [0, 1]")
    if mode not in ("qdrop", "rinp", "qinp"):
        raise ValueError("mode must be one of 'qdrop', 'rinp', or 'qinp'")
    if dis_mode not in ("q", "qf"):
        raise ValueError("dis_mode must be 'q' or 'qf'")
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise ValueError("model must have at least one parameter to determine its device")
    selected_device = torch.device(device)
    forward_batch = batch_size if microbatch_size is None else int(microbatch_size)
    if forward_batch <= 0:
        raise ValueError("microbatch_size must be positive")
    model.to(selected_device)
    was_training = model.training
    model.eval()
    unit_list = list(units)
    if not unit_list:
        raise ValueError("at least one reconstruction unit is required")
    for unit in unit_list:
        if not hasattr(unit, "name") or not hasattr(unit, "module") or not hasattr(unit, "is_head"):
            raise TypeError("units must expose name, module, and is_head")

    temp_dir = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(prefix="fima-q-recon-", dir=cache_dir)
        cache_path = temp_dir.name
    else:
        cache_path = None
    pending_adaround = []
    try:
        expected_ids, teacher_by_id = _teacher_cache(
            model, unit_list, loader, forward_batch, selected_device, temperature
        )
        for unit_index, _unit in enumerate(unit_list):
            pending_adaround.extend(_reconstruct_unit(
                model, unit_list, unit_index, loader, expected_ids,
                teacher_by_id, cache_path, selected_device, iterations,
                batch_size, forward_batch, rank, temperature, drop_prob,
                fisher_batch_size, mode, dis_mode, p1, p2,
            ))
        _harden_adaround_quantizers(pending_adaround)
        pending_adaround = []
        _set_model_modes(model, unit_list, range(len(unit_list)))
        return model
    finally:
        if pending_adaround:
            for child, original_quantizer, adaptive_quantizer in pending_adaround:
                adaptive_quantizer.soft_targets = False
                child.w_quantizer = original_quantizer
            _set_model_modes(model, unit_list)
        model.train(was_training)
        if temp_dir is not None:
            temp_dir.cleanup()
