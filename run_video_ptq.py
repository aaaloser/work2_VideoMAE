import argparse
import json
import logging
import os
from hashlib import sha256
from pathlib import Path

import numpy as np
import timm
import torch

from engine_for_finetuning import final_test, merge
from video_ptq.adapters.videomae import adapt_videomae, build_reference
from video_ptq.calibration import calibrate
from video_ptq.checkpoint import load_quantized, save_quantized
from video_ptq.config import STEM_POLICIES, QuantPolicy
from video_ptq.data import DATASET_CLASSES, build_calibration_source, build_evaluation_loader


ROOT = Path(__file__).resolve().parent
PROTOCOL_FIELDS = (
    'model', 'data_set', 'nb_classes', 'num_frames', 'tubelet_size', 'sampling_rate',
    'input_size', 'short_side_size', 'test_num_segment', 'test_num_crop', 'use_cls',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='FIMA-Q fake quantization for finetuned VideoMAE')
    parser.add_argument('phase', choices=('full', 'raw-check', 'calibrate', 'reconstruct', 'evaluate'),
                        help='Use full to calibrate, reconstruct, and evaluate both checkpoints')
    parser.add_argument('--run', help='Run identifier under out_log and out_checkpoint')
    parser.add_argument('--data-set', choices=DATASET_CLASSES, default='UCF101')
    parser.add_argument('--data-path', type=Path, help='Directory with train.csv and test.csv (defaults to UCF101 split 1)')
    parser.add_argument('--finetune', type=Path, help='Source checkpoint (defaults to local UCF101 ViT-B weights)')
    parser.add_argument('--model-key', default='module|model|state_dict')
    parser.add_argument('--model', default='vit_base_patch16_224')
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--tubelet-size', type=int, default=2)
    parser.add_argument('--sampling-rate', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--short-side-size', type=int, default=224)
    parser.add_argument('--test-num-segment', type=int, help='Defaults to 2 for SSV2 and 5 for UCF101/Kinetics-400')
    parser.add_argument('--test-num-crop', type=int, default=3)
    parser.add_argument('--use-cls', action='store_true')
    parser.add_argument('--drop-path', type=float, default=0.1)
    parser.add_argument('--stem', choices=STEM_POLICIES, default='a4',
                        help='Activation policy for the W4 tubelet projection; reference bypasses activation Q/DQ')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--calib-size', type=int, default=128)
    parser.add_argument('--optim-size', type=int, default=1024)
    parser.add_argument('--forward-batch-size', type=int, default=1, help='Clips per device microbatch')
    parser.add_argument('--eq-n', type=int, default=128)
    parser.add_argument('--search-round', type=int, default=3)
    parser.add_argument('--candidate-chunk', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--recon-batch-size', type=int, default=32, help='Clips per reconstruction optimizer step')
    parser.add_argument('--fisher-batch-size', type=int, default=32, help='Logical Fisher KL batch denominator')
    parser.add_argument('--rank', type=int, default=5, help='Fisher target rank')
    parser.add_argument('--temperature', type=float, default=20)
    parser.add_argument('--drop-prob', type=float, default=0.5, help='QDrop probability of retaining quantization')
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--checkpoint-stage', choices=('calibrated', 'reconstructed'), default='reconstructed',
                        help='Checkpoint to evaluate')
    args = parser.parse_args(argv)
    if args.data_set == 'UCF101':
        if args.data_path is None:
            args.data_path = ROOT / 'annotations' / 'ucf101_split1'
        if args.finetune is None and args.model == 'vit_base_patch16_224':
            args.finetune = ROOT / 'pretrained' / 'VideoMAE_ViT-B_16x1_224_UCF101.pth'
    if args.data_path is None or args.finetune is None:
        parser.error('Provide --data-path and --finetune where no local default is available')
    if args.run is None:
        dataset_name = args.data_set.lower().replace('-', '')
        model_name = 'vitb' if args.model == 'vit_base_patch16_224' else args.model
        stem_name = 'w4a4' if args.stem == 'a4' else f'stem_{args.stem}'
        args.run = f'{dataset_name}_{model_name}_fimaq_{stem_name}'
    if args.test_num_segment is None:
        args.test_num_segment = 2 if args.data_set == 'SSV2' else 5
    if not args.run or Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('--run must be a single path component')
    positive = (
        'num_frames', 'tubelet_size', 'sampling_rate', 'input_size', 'short_side_size', 'calib_size',
        'optim_size', 'forward_batch_size', 'recon_batch_size', 'fisher_batch_size', 'eval_batch_size',
        'iterations', 'rank',
    )
    if any(getattr(args, field) < 1 for field in positive) or args.workers < 0:
        parser.error('Geometry, sample counts, iterations, rank and batch sizes must be positive')
    if args.test_num_segment < 2 or args.test_num_crop < 2:
        parser.error('The existing test protocol needs at least two temporal and spatial positions')
    if args.data_set == 'SSV2' and args.test_num_segment != 2:
        parser.error('SSV2 test sampling uses exactly two interleaved temporal positions')
    if args.short_side_size != args.input_size or args.num_frames % args.tubelet_size:
        parser.error('Use matching resize/crop sizes and a frame count divisible by the tubelet size')
    if args.phase in ('reconstruct', 'full') and args.iterations < args.rank:
        parser.error('Reconstruction needs at least rank iterations')
    args.nb_classes = DATASET_CLASSES[args.data_set]
    return args


def file_digest(path):
    digest = sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_for(args):
    return {
        'protocol': {name: getattr(args, name) for name in PROTOCOL_FIELDS},
        'finetune_sha256': file_digest(args.finetune),
        'test_annotations_sha256': file_digest(args.data_path / 'test.csv'),
        'torch_version': torch.__version__, 'timm_version': timm.__version__,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'seed': args.seed,
    }


def verify_reference(metadata, expected):
    for key in ('protocol', 'finetune_sha256', 'test_annotations_sha256'):
        if metadata.get(key) != expected[key]:
            raise ValueError(f'Checkpoint {key} differs from the requested reference model or evaluation protocol')


def clip_source_versions(views):
    versions = []
    for view in views:
        stat = Path(view['path']).stat()
        versions.append((stat.st_size, stat.st_mtime_ns))
    return versions


def select_device(args):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    device = torch.device(args.device)
    if world_size > 1 and (args.phase != 'evaluate' or device.type != 'cuda'):
        raise ValueError('Distributed execution is available for CUDA evaluation')
    if device.type == 'cuda':
        os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this environment. Use --device cpu for CPU verification')
        if world_size > 1:
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device.index if device.index is not None else 0)
    return device, rank, world_size


@torch.no_grad()
def check_raw(reference, backbone, batch, device, *, amp=False):
    clips, _ = batch
    clips = clips.to(device)
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        if device.type == 'cuda' and not amp:
            # TF32 rounds differently after an otherwise equivalent affine fold.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with torch.cuda.amp.autocast(enabled=amp):
            expected, actual = reference(clips), backbone.model(clips)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
    maximum_error = float((expected - actual).abs().max())
    if not torch.allclose(expected, actual, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f'Raw VideoMAE parity failed with autocast={amp}, max abs error={maximum_error:.8g}')
    logging.info('Raw VideoMAE parity, autocast=%s, logits=%s, max abs error=%.8g',
                 amp, tuple(actual.shape), maximum_error)


def check_view_coverage(output, world_size, expected_views, args):
    by_video = {}
    rows = 0
    views_per_video = {(segment, crop) for segment in range(args.test_num_segment)
                       for crop in range(args.test_num_crop)}
    for rank in range(world_size):
        with (output / f'{rank}.txt').open() as predictions:
            next(predictions)
            for row in predictions:
                name, _, rest = row.partition(' [')
                _, _, fields = rest.partition(']')
                _, segment, crop = map(int, fields.split())
                by_video.setdefault(name, set()).add((segment, crop))
                rows += 1
    unique = sum(map(len, by_video.values()))
    expected_padding = (-expected_views) % world_size
    if unique != expected_views or rows - unique != expected_padding:
        raise RuntimeError(f'Expected {expected_views} unique views and {expected_padding} padding rows, '
                           f'got {unique} and {rows - unique}')
    if any(views != views_per_video for views in by_video.values()):
        raise RuntimeError('At least one video has an incomplete evaluation view grid')
    return dict(videos=len(by_video), views=unique, padding_rows=rows - unique)


def evaluate(backbone, args, device, rank, world_size):
    output = ROOT / 'out_log' / args.run / args.checkpoint_stage
    output.mkdir(parents=True, exist_ok=True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    loader = build_evaluation_loader(args, batch_size=args.eval_batch_size, workers=args.workers,
                                     rank=rank, world_size=world_size)
    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl')
    try:
        final_test(loader, backbone.model, device, output / f'{rank}.txt')
        if world_size > 1:
            torch.distributed.barrier()
        if rank == 0:
            stats = check_view_coverage(output, world_size, len(loader.dataset), args)
            top1, top5 = merge(str(output), world_size)
            stats.update({'Final top-1': top1, 'Final Top-5': top5})
            (output / 'metrics.json').write_text(json.dumps(stats, indent=2) + '\n')
            logging.info('Video-level Top-1 %.4f%%, Top-5 %.4f%%, %d videos, %d unique views',
                         top1, top5, stats['videos'], stats['views'])
    finally:
        if world_size > 1:
            torch.distributed.destroy_process_group()


def run_phase(args):
    device, rank, world_size = select_device(args)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    reference = build_reference(args)
    backbone = adapt_videomae(reference, QuantPolicy(stem=args.stem))
    backbone.model.to(device)
    logging.info('Phase %s, dataset %s, stem %s, quantized operators %d, device %s',
                 args.phase, args.data_set, args.stem, len(backbone.targets), device)

    checkpoint_dir = ROOT / 'out_checkpoint' / args.run
    cache_dir = checkpoint_dir / 'cache'
    if args.phase == 'raw-check':
        source = build_calibration_source(args, size=args.calib_size, seed=args.seed,
                                          batch_size=args.forward_batch_size, workers=args.workers)
        reference.to(device)
        batch = next(iter(source.loader))
        check_raw(reference, backbone, batch, device)
        if device.type == 'cuda':
            check_raw(reference, backbone, batch, device, amp=True)
        return

    metadata = metadata_for(args)
    if args.phase == 'evaluate':
        del reference
        stored = load_quantized(checkpoint_dir / f'{args.checkpoint_stage}.pth', backbone,
                                expected_stage=args.checkpoint_stage)
        verify_reference(stored, metadata)
        evaluate(backbone, args, device, rank, world_size)
        return

    if args.phase == 'calibrate':
        source = build_calibration_source(args, size=args.calib_size, seed=args.seed,
                                          batch_size=args.forward_batch_size, workers=args.workers,
                                          cache_dir=cache_dir / 'clips')
        versions = clip_source_versions(source.manifest)
        calibrate(backbone, source.loader, device=device, eq_n=args.eq_n, search_round=args.search_round,
                  candidate_chunk=args.candidate_chunk, cache_dir=cache_dir)
        if clip_source_versions(source.manifest) != versions:
            raise RuntimeError('A calibration video changed while the loader was being replayed')
        reference.to(device)
        for target in backbone.targets:
            target.module.mode = 'raw'
        check_raw(reference, backbone, next(iter(source.loader)), device)
        for target in backbone.targets:
            target.module.mode = 'quant_forward'
        del reference
        metadata['calibration'] = dict(views=source.manifest, source_versions=versions, eq_n=args.eq_n,
                                       search_round=args.search_round, candidate_chunk=args.candidate_chunk,
                                       batch_size=args.forward_batch_size)
        path = checkpoint_dir / 'calibrated.pth'
        save_quantized(path, backbone, stage='calibrated', metadata=metadata)
    else:
        from video_ptq.reconstruction import reconstruct

        del reference
        metadata_calib = load_quantized(checkpoint_dir / 'calibrated.pth', backbone, expected_stage='calibrated')
        verify_reference(metadata_calib, metadata)
        source = build_calibration_source(args, size=args.optim_size, seed=args.seed,
                                          batch_size=args.forward_batch_size, workers=args.workers,
                                          cache_dir=cache_dir / 'clips')
        calibrated_views = metadata_calib['calibration']['views']
        if args.seed != metadata_calib['seed'] or source.manifest[:len(calibrated_views)] != calibrated_views:
            raise ValueError('Reconstruction clips must extend the calibrated view manifest and reuse its seed')
        versions = clip_source_versions(source.manifest)
        calibrated_versions = metadata_calib['calibration'].get('source_versions')
        if calibrated_versions is not None and versions[:len(calibrated_views)] != calibrated_versions:
            raise RuntimeError('A calibration video changed before reconstruction')
        reconstruct(backbone.model, backbone.units, source.loader, device=device, cache_dir=cache_dir,
                    iterations=args.iterations, batch_size=args.recon_batch_size, rank=args.rank,
                    temperature=args.temperature, drop_prob=args.drop_prob,
                    fisher_batch_size=args.fisher_batch_size, microbatch_size=args.forward_batch_size)
        if clip_source_versions(source.manifest) != versions:
            raise RuntimeError('A reconstruction video changed while the loader was being replayed')
        metadata['calibration'] = metadata_calib['calibration']
        metadata['reconstruction'] = dict(views=source.manifest, source_versions=versions,
                                          iterations=args.iterations, rank=args.rank,
                                          temperature=args.temperature, drop_prob=args.drop_prob,
                                          batch_size=args.recon_batch_size, fisher_batch_size=args.fisher_batch_size,
                                          forward_batch_size=args.forward_batch_size)
        path = checkpoint_dir / 'reconstructed.pth'
        save_quantized(path, backbone, stage='reconstructed', metadata=metadata)
    logging.info('Saved %s', path)
    if device.type == 'cuda':
        logging.info('Peak CUDA allocated %.3f GiB, reserved %.3f GiB',
                     torch.cuda.max_memory_allocated(device) / 2**30,
                     torch.cuda.max_memory_reserved(device) / 2**30)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.phase != 'full':
        run_phase(args)
        return

    for phase, checkpoint_stage in (
        ('calibrate', 'calibrated'),
        ('reconstruct', 'reconstructed'),
        ('evaluate', 'calibrated'),
        ('evaluate', 'reconstructed'),
    ):
        run_phase(argparse.Namespace(**{**vars(args), 'phase': phase, 'checkpoint_stage': checkpoint_stage}))


if __name__ == '__main__':
    main()
