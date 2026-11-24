import unittest

import torch
import torch.nn.functional as F

from video_ptq.adapter import ReconstructionUnit
from video_ptq.linear import QuantLinear
from video_ptq.reconstruction import (
    dplr_terms,
    fisher_kl_loss,
    head_kl,
    normalized_dplr_loss,
    reconstruct,
    teacher_probabilities,
)


class ReconstructionMathTest(unittest.TestCase):
    def test_dplr_terms_match_low_rank_and_diagonal_formula(self):
        # E has rows [1, 2] and [2, 1]. G is identity, and inverse_B is
        # diag(2, 1), so the per-sample low-rank terms are 6 and 9.
        error = torch.tensor([[1.0, -2.0], [-2.0, 1.0]])
        raw_grad = torch.eye(2)
        inverse_B = torch.tensor([[2.0, 0.0], [0.0, 1.0]])

        low_rank, diagonal = dplr_terms(error, raw_grad, inverse_B)

        self.assertAlmostEqual(float(low_rank), 7.5, places=6)
        self.assertAlmostEqual(float(diagonal), 1.25, places=6)

    def test_normalized_dplr_loss_uses_both_reference_terms_and_weights(self):
        error = torch.tensor([[1.0, -2.0], [-2.0, 1.0]])
        raw_grad = torch.eye(2)
        inverse_B = torch.tensor([[2.0, 0.0], [0.0, 1.0]])

        loss = normalized_dplr_loss(
            error,
            raw_grad,
            inverse_B,
            reference=(3.0, 0.5),
            p1=2.0,
            p2=4.0,
        )

        # (7.5 / 3) * 2 + (1.25 / 0.5) * 4 = 15.
        self.assertAlmostEqual(float(loss), 15.0, places=6)

    def test_normalized_dplr_loss_rejects_zero_reference_components(self):
        with self.assertRaisesRegex(ValueError, "reference term is zero"):
            normalized_dplr_loss(
                torch.ones(2, 2),
                torch.eye(2),
                torch.eye(2),
                reference=(0.0, 1.0),
            )

    def test_fisher_kl_microchunks_keep_the_logical_batch_denominator(self):
        student = torch.tensor(
            [[1.2, -0.4, 0.7], [-0.2, 0.3, 0.9], [0.8, 0.1, -0.5]],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]]
        )
        temperature = 2.5

        reference = F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            teacher,
            reduction="batchmean",
        )
        reference_grad = torch.autograd.grad(reference, student)[0]

        chunked_student = student.detach().clone().requires_grad_(True)
        chunked = sum(
            fisher_kl_loss(
                chunked_student[i:i + 1],
                teacher[i:i + 1],
                temperature=temperature,
                logical_batch_size=3,
            )
            for i in range(3)
        )
        chunked_grad = torch.autograd.grad(chunked, chunked_student)[0]

        torch.testing.assert_allclose(chunked, reference, atol=1e-7, rtol=1e-6)
        torch.testing.assert_allclose(chunked_grad, reference_grad, atol=1e-7, rtol=1e-6)

    def test_fisher_kl_accepts_per_sample_denominators_for_partial_groups(self):
        student = torch.tensor([[0.7, -0.1], [0.2, 0.8], [-0.4, 0.6]])
        teacher = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
        chunked = fisher_kl_loss(
            student, teacher, temperature=3.0, logical_batch_size=[2, 2, 1]
        )
        first_group = F.kl_div(
            F.log_softmax(student[:2] / 3.0, dim=-1),
            teacher[:2],
            reduction="batchmean",
        )
        final_group = F.kl_div(
            F.log_softmax(student[2:] / 3.0, dim=-1),
            teacher[2:],
            reduction="batchmean",
        )
        torch.testing.assert_allclose(
            chunked, first_group + final_group, atol=1e-7, rtol=1e-6
        )

    def test_teacher_uses_temperature_and_head_uses_original_logit_kl(self):
        teacher_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        student_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        probs = teacher_probabilities(teacher_logits, temperature=2.0)
        expected_probs = torch.tensor(
            [[0.7310586, 0.2689414], [0.2689414, 0.7310586]]
        )
        torch.testing.assert_allclose(probs, expected_probs, atol=1e-6, rtol=1e-6)

        expected_head_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )
        self.assertAlmostEqual(float(head_kl(student_logits, teacher_logits)),
                               float(expected_head_loss), places=7)


class _TinyQuantizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        source = torch.nn.Linear(3, 2)
        self.proj = QuantLinear(source, w_bit=4, a_bit=4, token_channel_wise=False)

    def forward(self, clips):
        return self.proj(clips)


class _TwoLayerQuantizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = QuantLinear(
            torch.nn.Linear(3, 3), w_bit=4, a_bit=4, token_channel_wise=False
        )
        self.head = QuantLinear(
            torch.nn.Linear(3, 2), w_bit=4, a_bit=4, token_channel_wise=False
        )

    def forward(self, clips):
        return self.head(self.proj(clips))


class ReconstructionIntegrationTest(unittest.TestCase):
    def test_reconstruct_runs_fisher_with_single_clip_forwards_and_restores_uniform_q(self):
        torch.manual_seed(73)
        model = _TinyQuantizedModel().eval()
        source = torch.nn.Linear(3, 2)
        source.load_state_dict({
            "weight": model.proj.weight.detach().clone(),
            "bias": model.proj.bias.detach().clone(),
        })
        clips = [torch.tensor([[0.3, -0.7, 0.5]]), torch.tensor([[-0.4, 0.2, 0.9]])]
        records = [((clip,), source(clip).detach()) for clip in clips]
        model.proj.calibrate(records, eq_n=16, search_round=1, candidate_chunk=4)
        units = [ReconstructionUnit("proj", model.proj)]
        loader = [(clips[0], ["view-a"]), (clips[1], ["view-b"])]
        state_keys = set(model.state_dict().keys())

        result = reconstruct(
            model,
            units,
            loader,
            device="cpu",
            iterations=1,
            batch_size=2,
            rank=1,
            temperature=2.0,
            drop_prob=0.5,
            fisher_batch_size=2,
            microbatch_size=1,
        )

        self.assertIs(result, model)
        self.assertEqual(model.proj.mode, "quant_forward")
        self.assertEqual(set(model.state_dict().keys()), state_keys)
        self.assertEqual(type(model.proj.w_quantizer).__name__, "UniformQuantizer")
        self.assertFalse(hasattr(model.proj.w_quantizer, "alpha"))
        self.assertFalse(model.proj.a_quantizer.training_mode)
        self.assertEqual(model.proj.a_quantizer.drop_prob, 1.0)
        self.assertTrue(torch.isfinite(model(torch.cat(clips))).all())

    def test_later_unit_raw_cache_uses_original_prefix_and_quant_input_uses_hard_adaround(self):
        torch.manual_seed(79)
        model = _TwoLayerQuantizedModel().eval()
        clips = [torch.tensor([[0.3, -0.7, 0.5]]), torch.tensor([[-0.4, 0.2, 0.9]])]
        with torch.no_grad():
            raw_hidden = [model.proj(clip).detach().clone() for clip in clips]
            raw_logits = [model.head(hidden).detach() for hidden in raw_hidden]
        model.proj.calibrate(
            [((clip,), hidden) for clip, hidden in zip(clips, raw_hidden)],
            eq_n=16,
            search_round=1,
            candidate_chunk=4,
        )
        model.head.calibrate(
            [((hidden,), logits) for hidden, logits in zip(raw_hidden, raw_logits)],
            eq_n=16,
            search_round=1,
            candidate_chunk=4,
        )
        initial_hidden = [model.proj(clip).detach().clone() for clip in clips]
        units = [
            ReconstructionUnit("proj", model.proj),
            ReconstructionUnit("head", model.head, is_head=True),
        ]
        loader = [(clip, ["view-{}".format(index)]) for index, clip in enumerate(clips)]
        observed = []

        def capture_head_input(module, args):
            observed.append((model.proj.mode, module.mode, args[0].detach().clone()))

        handle = model.head.register_forward_pre_hook(capture_head_input)
        try:
            reconstruct(
                model,
                units,
                loader,
                device="cpu",
                iterations=1,
                batch_size=2,
                rank=1,
                temperature=2.0,
                drop_prob=1.0,
                fisher_batch_size=2,
                microbatch_size=1,
            )
        finally:
            handle.remove()

        raw_replays = [item for item in observed
                       if item[0] == "raw" and item[1] == "raw"]
        quantized_prefix_replays = [item for item in observed
                                    if item[0] == "quant_forward" and item[1] == "raw"]
        self.assertGreaterEqual(len(raw_replays), 4)
        self.assertGreaterEqual(len(quantized_prefix_replays), 4)
        for observed_item, expected in zip(raw_replays[-2:], initial_hidden):
            torch.testing.assert_allclose(observed_item[2], expected, atol=1e-6, rtol=1e-6)
        final_quantized_hidden = [model.proj(clip).detach() for clip in clips]
        for observed_item, expected in zip(quantized_prefix_replays[-2:], final_quantized_hidden):
            torch.testing.assert_allclose(observed_item[2], expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
