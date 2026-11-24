import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


class _CapturedOutput(Exception):
    """Stop a forward after the calibration target has executed."""


class _Records:
    def __init__(self, root):
        self.root = Path(root)
        self.count = 0

    def append(self, inputs, output):
        record = (
            tuple(x.detach().to(device='cpu', copy=True) for x in inputs),
            output.detach().to(device='cpu', copy=True),
        )
        torch.save(record, self.root / f'{self.count}.pt')
        self.count += 1

    def __iter__(self):
        for index in range(self.count):
            yield torch.load(self.root / f'{index}.pt', map_location='cpu')


@torch.no_grad()
def calibrate(backbone, loader, *, device, eq_n=128, search_round=3, candidate_chunk=1, cache_dir=None):
    if eq_n < 16 or eq_n % 16 or search_round < 1 or candidate_chunk < 1:
        raise ValueError('Calibration needs a candidate count divisible by 16 and positive search rounds/chunk')
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    backbone.model.to(device).eval()
    for target in backbone.targets:
        target.module.mode = 'raw'
    expected_ids = None
    for target in backbone.targets:
        logging.info('Calibrating %s', target.name)
        with TemporaryDirectory(prefix='operator-', dir=cache_dir) as storage:
            records = _Records(storage)

            def capture(module, inputs, output):
                records.append(inputs, output)
                raise _CapturedOutput

            handle = target.module.register_forward_hook(capture)
            observed_ids = []
            try:
                for clips, ids in loader:
                    observed_ids.extend(ids)
                    try:
                        backbone.model(clips.to(device))
                    except _CapturedOutput:
                        continue
                    raise RuntimeError(f'{target.name} was not executed during calibration')
            finally:
                handle.remove()
            if not observed_ids or len(set(observed_ids)) != len(observed_ids):
                raise ValueError('Calibration needs non-empty, unique clip ids')
            if expected_ids is not None and observed_ids != expected_ids:
                raise RuntimeError('Calibration view identity or order changed during replay')
            expected_ids = observed_ids
            target.module.calibrate(
                records, eq_n=eq_n, search_round=search_round,
                candidate_chunk=candidate_chunk, prev_layer=target.prev_layer,
            )
            target.module.mode = 'raw'

    for target in backbone.targets:
        target.module.mode = 'quant_forward'
