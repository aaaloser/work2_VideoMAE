import pytest

import run_video_ptq as cli


def test_full_command_uses_local_ucf101_defaults_and_evaluates_both_checkpoints(monkeypatch):
    phases = []

    def record_phase(args):
        phases.append((args.phase, args.checkpoint_stage))
        assert args.device == 'cuda:2'
        assert args.run == 'ucf101_vitb_fimaq_w4a4'
        assert args.data_path == cli.ROOT / 'annotations' / 'ucf101_split1'
        assert args.finetune == cli.ROOT / 'pretrained' / 'VideoMAE_ViT-B_16x1_224_UCF101.pth'
        assert (args.calib_size, args.optim_size, args.iterations) == (128, 1024, 20000)

    monkeypatch.setattr(cli, 'run_phase', record_phase, raising=False)
    cli.main(['full', '--device', 'cuda:2'])

    assert phases == [
        ('calibrate', 'calibrated'),
        ('reconstruct', 'reconstructed'),
        ('evaluate', 'calibrated'),
        ('evaluate', 'reconstructed'),
    ]


def test_full_command_stops_if_a_required_stage_fails(monkeypatch):
    phases = []

    def fail_calibration(args):
        phases.append(args.phase)
        raise RuntimeError('calibration failed')

    monkeypatch.setattr(cli, 'run_phase', fail_calibration, raising=False)
    with pytest.raises(RuntimeError, match='calibration failed'):
        cli.main(['full', '--device', 'cpu'])

    assert phases == ['calibrate']
