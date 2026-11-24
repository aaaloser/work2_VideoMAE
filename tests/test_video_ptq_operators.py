"""Small CPU reference tests for the FIMA-Q fake-quant operators."""

import torch
from torch import nn
from torch.nn import functional as F

from video_ptq.conv3d import QuantConv3d
from video_ptq.linear import QuantLinear
from video_ptq.matmul import QuantMatMul
from video_ptq.quantizers import AdaRoundQuantizer, UniformQuantizer
import video_ptq._calibration as calibration


def _assert_close(actual, expected, atol=1e-6, rtol=1e-5):
    torch.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _records_linear(source, batches):
    return [((x,), F.linear(x, source.weight, source.bias).detach()) for x in batches]


def test_uniform_quantizer_matches_fima_q_round_clip_and_zero_point():
    symmetric = UniformQuantizer(n_bits=4, symmetric=True)
    symmetric.scale = nn.Parameter(torch.tensor(0.5))
    symmetric.inited = True
    x = torch.tensor([-5.0, -4.25, -0.25, 0.25, 3.75, 5.0])
    expected = torch.round(x / 0.5).clamp(-8, 7) * 0.5
    _assert_close(symmetric(x), expected)

    asymmetric = UniformQuantizer(n_bits=4, symmetric=False)
    asymmetric.scale = nn.Parameter(torch.tensor(0.25))
    asymmetric.zero_point = nn.Parameter(torch.tensor(3.6))
    asymmetric.inited = True
    expected = (torch.round(x / 0.25) + torch.round(torch.tensor(3.6))).clamp(0, 15)
    expected = (expected - torch.round(torch.tensor(3.6))) * 0.25
    _assert_close(asymmetric(x), expected)


def test_memmap_quantiles_match_pooled_torch_quantile(tmp_path, monkeypatch):
    class ReplayableRecords(object):
        def __init__(self, rows):
            self.rows = rows
            self.root = str(tmp_path)

        def __iter__(self):
            return iter(self.rows)

    monkeypatch.setattr(calibration, "_EXACT_QUANTILE_LIMIT", 1)
    torch.manual_seed(31)
    rows = []
    for batch in (1, 2):
        x = torch.randn(batch, 3, 7, 5)
        rows.append(((x,), torch.zeros(batch, 1)))
    records = ReplayableRecords(rows)
    probabilities = [0.0, 0.1, 0.5, 0.9, 1.0]
    got = calibration.quantile_summary(records, 1, 0, probabilities, grouping="head")
    pooled = torch.cat([row[0][0] for row in rows], dim=0).transpose(0, 1).contiguous().reshape(3, -1)
    expected = torch.quantile(pooled, torch.tensor(probabilities), dim=1)
    _assert_close(got, expected, atol=1e-6, rtol=1e-6)
    assert list(tmp_path.iterdir()) == []


def test_adaround_uses_learned_hard_and_soft_targets_and_preserves_grid():
    uq = UniformQuantizer(n_bits=4, symmetric=False)
    uq.scale = nn.Parameter(torch.tensor([[0.25, 0.5]]))
    uq.zero_point = nn.Parameter(torch.tensor([[7.0, 7.0]]))
    uq.inited = True
    weight = torch.tensor([[0.13, 0.62]])
    quantizer = AdaRoundQuantizer(uq, weight)

    quantizer.alpha.data.fill_(1.0)
    hard = quantizer(weight)
    _assert_close(hard, torch.tensor([[0.25, 1.0]]))
    _assert_close(quantizer.get_hard_value(weight), hard)

    quantizer.soft_targets = True
    soft = quantizer(weight)
    assert torch.isfinite(soft).all()
    assert not torch.equal(soft, hard)
    soft.sum().backward()
    assert quantizer.alpha.grad is not None
    assert quantizer.scale is not uq.scale
    assert quantizer.zero_point is not uq.zero_point
    assert not quantizer.scale.requires_grad
    assert not quantizer.zero_point.requires_grad


def test_linear_raw_quant_and_grouped_token_calibration():
    torch.manual_seed(4)
    source = nn.Linear(4, 6, bias=True)
    batches = [torch.randn(1, 5, 4), torch.randn(1, 5, 4) * 0.7]
    layer = QuantLinear(source, w_bit=4, a_bit=4, n_groups=3, token_channel_wise=True)

    x = batches[0]
    _assert_close(layer(x), source(x))
    assert layer.weight_view().shape == (3, 2, 4)
    assert torch.equal(layer.weight_view().reshape_as(layer.weight), layer.weight)

    layer.calibrate(_records_linear(source, batches), eq_n=32, search_round=1, candidate_chunk=3)
    assert layer.calibrated
    assert layer.w_quantizer.inited and layer.a_quantizer.inited
    assert layer.w_quantizer.scale.shape == (3, 2, 1)
    assert layer.w_quantizer.zero_point.shape == (3, 2, 1)
    assert layer.a_quantizer.scale.shape == (1, 5, 1)
    assert layer.a_quantizer.zero_point.numel() == 1

    layer.mode = "quant_forward"
    out = layer(x)
    assert out.shape == (1, 5, 6)
    assert torch.isfinite(out).all()
    expected = F.linear(layer.a_quantizer(x), layer.w_quantizer(layer.weight_view()).reshape_as(layer.weight), layer.bias)
    _assert_close(out, expected)


def test_linear_prev_layer_affine_reparameterization_preserves_raw_function():
    torch.manual_seed(5)
    prev = nn.LayerNorm(4)
    source = nn.Linear(4, 6, bias=True)
    before_prev = nn.LayerNorm(4)
    before_prev.load_state_dict(prev.state_dict())
    before_linear = nn.Linear(4, 6, bias=True)
    before_linear.load_state_dict(source.state_dict())
    raw_inputs = [torch.randn(1, 6, 4), torch.randn(1, 6, 4) * 1.2]
    post_norm = [before_prev(x).detach() for x in raw_inputs]
    records = _records_linear(before_linear, post_norm)
    layer = QuantLinear(source, w_bit=4, a_bit=4, token_channel_wise=True)

    layer.calibrate(records, eq_n=32, search_round=1, candidate_chunk=4, prev_layer=prev)

    for x in raw_inputs:
        expected = before_linear(before_prev(x))
        got = layer(prev(x))
        _assert_close(got, expected, atol=2e-5, rtol=2e-5)
    assert layer.a_quantizer.scale.shape == (1, 6, 1)


def test_matmul_raw_and_per_head_token_quantization():
    torch.manual_seed(6)
    layer = QuantMatMul(num_heads=2, a_bit=4, token_channel_wise=True, head_channel_wise=True)
    records = []
    for batch in (1, 2):
        a = torch.randn(batch, 2, 4, 3)
        b = torch.randn(batch, 2, 3, 4)
        records.append(((a, b), (a @ b).detach()))

    a, b = records[0][0]
    _assert_close(layer(a, b), a @ b)
    layer.calibrate(records, eq_n=32, search_round=1, candidate_chunk=3)
    assert layer.calibrated
    assert layer.A_quantizer.inited and layer.B_quantizer.inited
    assert layer.A_quantizer.scale.shape == (1, 2, 4, 1)
    assert layer.B_quantizer.scale.shape == (1, 2, 1, 4)
    assert layer.A_quantizer.zero_point.shape == (1, 2, 1, 1)
    assert layer.B_quantizer.zero_point.shape == (1, 2, 1, 1)

    layer.mode = "quant_forward"
    _assert_close(layer(a, b), layer.A_quantizer(a) @ layer.B_quantizer(b))


def test_conv3d_copies_reference_and_calibrates_in_native_layout():
    torch.manual_seed(7)
    source = nn.Conv3d(2, 4, kernel_size=(1, 2, 2), stride=(1, 1, 1), padding=0, bias=True)
    layer = QuantConv3d(source, w_bit=4, a_bit=4)
    x_batches = [torch.randn(1, 2, 3, 4, 4), torch.randn(1, 2, 3, 4, 4) * 0.8]
    records = [((x,), source(x).detach()) for x in x_batches]

    _assert_close(layer(x_batches[0]), source(x_batches[0]))
    layer.calibrate(records, eq_n=32, search_round=1, candidate_chunk=2)
    assert layer.calibrated
    assert layer.w_quantizer.scale.shape == (4, 1, 1, 1, 1)
    assert layer.w_quantizer.zero_point.shape == (4, 1, 1, 1, 1)
    layer.mode = "quant_forward"
    out = layer(x_batches[0])
    expected = F.conv3d(
        layer.a_quantizer(x_batches[0]),
        layer.w_quantizer(layer.weight),
        layer.bias,
        layer.stride,
        layer.padding,
        layer.dilation,
        layer.groups,
    )
    _assert_close(out, expected)
