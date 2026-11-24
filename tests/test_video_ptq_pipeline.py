import tempfile
import unittest
from pathlib import Path

import torch

from modeling_finetune import VisionTransformer
from video_ptq.adapters.videomae import adapt_videomae
from video_ptq.calibration import calibrate
from video_ptq.checkpoint import load_quantized, save_quantized
from video_ptq.config import QuantPolicy


class VideoPTQPipelineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(70)
        self.model = VisionTransformer(
            img_size=16, patch_size=8, embed_dim=8, depth=2, num_heads=2,
            all_frames=4, tubelet_size=2, num_classes=5, mlp_ratio=2,
            qkv_bias=True, init_values=0.1, init_scale=1,
        ).eval()
        for block in self.model.blocks:
            block.attn.q_bias.data.normal_()
            block.attn.v_bias.data.normal_()
        self.clips = torch.randn(2, 3, 4, 16, 16)

    @torch.no_grad()
    def test_raw_wrap_preserves_bias_scale_and_block_execution(self):
        adapted = adapt_videomae(self.model, QuantPolicy())
        inputs = []

        def capture_qk(module, args):
            inputs.append(args)

        attn = adapted.model.blocks[0].attn
        hook = attn.matmul1.register_forward_pre_hook(capture_qk)
        try:
            logits = adapted.model(self.clips)
        finally:
            hook.remove()
        self.assertTrue(torch.equal(logits, self.model(self.clips)))
        self.assertEqual(len(inputs), 1)
        self.assertEqual(tuple(inputs[0][0].shape), (2, 2, 8, 4))
        self.assertEqual(tuple(inputs[0][1].shape), (2, 2, 4, 8))
        original = self.model.blocks[0].attn
        torch.testing.assert_allclose(attn.qkv.bias[:8], original.q_bias)
        self.assertTrue(torch.equal(attn.qkv.bias[8:16], torch.zeros(8)))
        self.assertEqual(len(adapted.targets), 14)
        self.assertEqual([unit.name for unit in adapted.units], ['patch_embed', 'blocks.0', 'blocks.1', 'head'])

    def test_calibrated_checkpoint_resizes_scales_and_restores_logits(self):
        self.model.blocks[0].attn.q_bias = None
        self.model.blocks[0].attn.v_bias = None
        reference_state = {name: parameter.clone() for name, parameter in self.model.state_dict().items()}
        adapted = adapt_videomae(self.model, QuantPolicy(stem='float'))
        batches = [(self.clips[:1], ['clip-a']), (self.clips[1:], ['clip-b'])]
        with tempfile.TemporaryDirectory() as folder:
            calibrate(adapted, batches, device=torch.device('cpu'), eq_n=32,
                      search_round=1, candidate_chunk=3, cache_dir=folder)
            self.assertTrue(all(target.module.calibrated for target in adapted.targets))
            path = Path(folder) / 'calibrated.pth'
            save_quantized(path, adapted, stage='calibrated', metadata={'num_frames': 4})
            restored = adapt_videomae(self.model, QuantPolicy(stem='float'))
            metadata = load_quantized(path, restored, expected_stage='calibrated')
            self.assertEqual(metadata, {'num_frames': 4})
            self.assertEqual(tuple(restored.model.blocks[0].attn.matmul1.A_quantizer.scale.shape), (1, 2, 8, 1))
            with torch.no_grad():
                original = self.model(self.clips)
                quantized = adapted.model(self.clips)
                self.assertTrue(torch.isfinite(quantized).all())
                self.assertFalse(torch.allclose(original, quantized))
                torch.testing.assert_allclose(restored.model(self.clips), quantized)
                for target in adapted.targets:
                    target.module.mode = 'raw'
                torch.testing.assert_allclose(adapted.model(self.clips), original, rtol=1e-4, atol=1e-5)
                for name, parameter in self.model.state_dict().items():
                    self.assertTrue(torch.equal(parameter, reference_state[name]), name)

    def test_reconstructed_checkpoint_round_trip_includes_tubelet_projection(self):
        from video_ptq.reconstruction import reconstruct

        adapted = adapt_videomae(self.model, QuantPolicy())
        batches = [(self.clips[:1], ['clip-a']), (self.clips[1:], ['clip-b'])]
        with tempfile.TemporaryDirectory() as folder:
            device = torch.device('cpu')
            calibrate(adapted, batches, device=device, eq_n=32, search_round=1,
                      candidate_chunk=3, cache_dir=folder)
            reconstruct(adapted.model, adapted.units, batches, device=device, cache_dir=folder,
                        iterations=2, batch_size=2, fisher_batch_size=2, rank=1, microbatch_size=1)
            path = Path(folder) / 'reconstructed.pth'
            save_quantized(path, adapted, stage='reconstructed', metadata={})
            restored = adapt_videomae(self.model, QuantPolicy())
            load_quantized(path, restored, expected_stage='reconstructed')
            with torch.no_grad():
                actual = restored.model(self.clips)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_allclose(actual, adapted.model(self.clips))

    def test_reference_stem_keeps_the_configured_a8_path_in_bypass(self):
        stem = adapt_videomae(self.model, QuantPolicy(stem='reference')).model.patch_embed.proj
        self.assertEqual(stem.a_quantizer.n_bits, 8)
        stem.calibrate([((self.clips,), stem(self.clips).detach())],
                       eq_n=16, search_round=1, candidate_chunk=2)
        stem.mode = 'quant_forward'
        expected = stem._conv_forward(self.clips, stem.w_quantizer(stem.weight), stem.bias)
        self.assertTrue(torch.equal(stem(self.clips), expected))


if __name__ == '__main__':
    unittest.main()
