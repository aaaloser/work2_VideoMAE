"""VideoMAE-backed calibration and evaluation data loaders for video PTQ."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler


DATASET_CLASSES = {"UCF101": 101, "Kinetics-400": 400, "SSV2": 174}
_CLIP_CACHE_VERSION = 1


def build_dataset(is_train, test_mode, args):
    """Import the repository builder lazily so calibration can be used in isolation."""
    from datasets import build_dataset as videomae_build_dataset

    return videomae_build_dataset(is_train=is_train, test_mode=test_mode, args=args)


def _read_path_column(csv_path: Path, *, allow_empty: bool = False) -> List[str]:
    """Read only the path field from VideoMAE's whitespace-delimited annotations."""
    paths = []
    with csv_path.open("r", encoding="utf-8") as annotation:
        for line_number, line in enumerate(annotation, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.rsplit(None, 1)
            # A normal VideoMAE row is ``<path> <label>``. Keeping the whole
            # row when there is no second field also supports path-only lists.
            path = fields[0] if len(fields) == 2 else line
            if not path:
                raise ValueError("empty video path in {}:{}".format(csv_path, line_number))
            if any(character.isspace() for character in path):
                raise ValueError(
                    "VideoMAE's native annotation parser does not support whitespace in paths: {}".format(
                        path
                    )
                )
            paths.append(path)
    if not paths and not allow_empty:
        raise ValueError("annotation has no video paths: {}".format(csv_path))
    return paths


def _path_identity(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.path.expanduser(path))))


def _video_stem(path: str) -> str:
    # Match the native VideoClsDataset / SSVideoClsDataset return value.
    return path.split("/")[-1].split(".")[0]


def _assert_unique_paths(paths: Sequence[str], annotation: Path) -> None:
    seen = set()
    for path in paths:
        identity = _path_identity(path)
        if identity in seen:
            raise ValueError("duplicate video path in {}: {}".format(annotation, path))
        seen.add(identity)


def _make_native_test_dataset(args, anno_path: str):
    """Create the repository's test-mode dataset with its original parameters."""
    if args.data_set == "SSV2":
        from ssv2 import SSVideoClsDataset

        dataset_class = SSVideoClsDataset
        dataset_options = {
            "clip_len": 1,
            "num_segment": args.num_frames,
        }
    elif args.data_set in ("UCF101", "Kinetics-400"):
        from kinetics import VideoClsDataset

        dataset_class = VideoClsDataset
        dataset_options = {
            "clip_len": args.num_frames,
            "frame_sample_rate": args.sampling_rate,
            "num_segment": 1,
        }
    else:
        raise NotImplementedError(
            "video PTQ data supports UCF101, Kinetics-400, and SSV2; got {!r}".format(
                args.data_set
            )
        )

    return dataset_class(
        anno_path=anno_path,
        data_path="/",
        mode="test",
        **dataset_options,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args,
    )


def _make_decode_fail_fast(native_dataset) -> None:
    """Turn the native dataset's silent random retry into a visible decode error."""
    native_loader = getattr(native_dataset, "loadvideo_decord", None)
    if not callable(native_loader):
        return

    def loadvideo_or_raise(sample, *args, **kwargs):
        frames = native_loader(sample, *args, **kwargs)
        if frames is None or len(frames) == 0:
            raise RuntimeError(
                "VideoMAE could not decode calibration video {!r}; refusing random fallback".format(
                    sample
                )
            )
        return frames

    # The dataset's test __getitem__ remains responsible for all decoding and
    # preprocessing. This guard only stops its training/evaluation retry loop
    # from silently switching the requested calibration sample.
    native_dataset.loadvideo_decord = loadvideo_or_raise


def _write_dummy_annotation(paths: Sequence[str], directory: Optional[Path]) -> str:
    """Write train paths with dummy labels for the native dataset's parser."""
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="video-ptq-paths-",
        suffix=".csv",
        dir=str(directory) if directory is not None else None,
        delete=False,
    )
    try:
        with handle:
            for path in paths:
                handle.write("{} 0\n".format(path))
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return handle.name


def _collate_calibration_batch(batch: Sequence[Tuple[torch.Tensor, str]]):
    clips, video_ids = zip(*batch)
    return torch.stack(clips, dim=0), list(video_ids)


class _SelectedTestViews(Dataset):
    """Replay selected entries through VideoMAE's native test-mode Dataset."""

    def __init__(
        self,
        native_dataset,
        manifest: Sequence[Dict[str, Any]],
        train_paths: Sequence[str],
        args,
        cache_dir: Optional[Path],
    ):
        self.native_dataset = native_dataset
        self.manifest = list(manifest)
        self.train_paths = list(train_paths)
        self.args = args
        self.cache_dir = cache_dir
        self.num_crops = int(args.test_num_crop)
        self.num_segments = int(args.test_num_segment)
        self._path_to_index = {path: index for index, path in enumerate(self.train_paths)}

        if len(getattr(native_dataset, "dataset_samples", ())) != len(self.train_paths):
            raise RuntimeError("native test dataset did not preserve the train path list")
        if len(getattr(native_dataset, "test_dataset", ())) != (
            len(self.train_paths) * self.num_segments * self.num_crops
        ):
            raise RuntimeError("native test dataset has an unexpected test-view grid")

        stem_paths: Dict[str, str] = {}
        for path in self.train_paths:
            stem = _video_stem(path)
            if stem in stem_paths and stem_paths[stem] != path:
                raise ValueError(
                    "native dataset returns basename IDs, which are ambiguous for {!r} and {!r}".format(
                        stem_paths[stem], path
                    )
                )
            stem_paths[stem] = path

    def __len__(self):
        return len(self.manifest)

    def _native_index(self, entry: Dict[str, Any]) -> int:
        video_index = self._path_to_index[entry["path"]]
        return (
            (int(entry["segment"]) * self.num_crops + int(entry["crop"]))
            * len(self.train_paths)
            + video_index
        )

    def _cache_file(self, entry: Dict[str, Any]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        path = Path(entry["path"])
        stat = self._source_stat(path)
        cache_spec = {
            "version": _CLIP_CACHE_VERSION,
            "path": _path_identity(str(path)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "dataset": self.args.data_set,
            "num_frames": int(self.args.num_frames),
            "sampling_rate": int(self.args.sampling_rate),
            "input_size": int(self.args.input_size),
            "short_side_size": int(self.args.short_side_size),
            "test_num_segment": self.num_segments,
            "test_num_crop": self.num_crops,
            "segment": int(entry["segment"]),
            "crop": int(entry["crop"]),
        }
        key = hashlib.sha256(
            json.dumps(cache_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / (key + ".pt")

    @staticmethod
    def _source_stat(path: Path):
        try:
            stat = path.stat()
        except OSError as error:
            raise FileNotFoundError("cannot access calibration video: {}".format(path)) from error
        if stat.st_size < 1024:
            raise ValueError("calibration video is smaller than 1 KiB: {}".format(path))
        return stat

    @staticmethod
    def _load_cached_clip(cache_file: Path) -> Optional[torch.Tensor]:
        if not cache_file.is_file():
            return None
        try:
            payload = torch.load(str(cache_file), map_location="cpu", weights_only=True)
        except TypeError:  # torch versions before the weights_only keyword
            payload = torch.load(str(cache_file), map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("clip"), torch.Tensor):
            raise RuntimeError("invalid decoded-clip cache entry: {}".format(cache_file))
        return payload["clip"]

    @staticmethod
    def _save_cached_clip(cache_file: Path, clip: torch.Tensor) -> None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = cache_file.with_name(
            "{}.{}.{}.tmp".format(cache_file.name, os.getpid(), threading.get_ident())
        )
        try:
            torch.save({"clip": clip.detach().cpu().contiguous()}, str(temp_file))
            os.replace(str(temp_file), str(cache_file))
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def __getitem__(self, index):
        entry = self.manifest[index]
        self._source_stat(Path(entry["path"]))
        cache_file = self._cache_file(entry)
        if cache_file is not None:
            cached_clip = self._load_cached_clip(cache_file)
            if cached_clip is not None:
                return cached_clip, entry["video_id"]

        native_index = self._native_index(entry)
        native_result = self.native_dataset[native_index]
        if not isinstance(native_result, (tuple, list)) or len(native_result) < 5:
            raise RuntimeError("native VideoMAE test dataset returned an unexpected sample")
        clip, _native_label, native_id, actual_segment, actual_crop = native_result[:5]

        expected_id = _video_stem(entry["path"])
        if str(native_id) != expected_id:
            raise RuntimeError(
                "native test dataset changed calibration identity from {!r} to {!r}; "
                "this usually indicates random decode fallback".format(expected_id, native_id)
            )
        if int(actual_segment) != int(entry["segment"]) or int(actual_crop) != int(entry["crop"]):
            raise RuntimeError(
                "native test dataset changed calibration view for {!r}: expected ({}, {}), "
                "got ({}, {}); this usually indicates random decode fallback".format(
                    entry["video_id"],
                    entry["segment"],
                    entry["crop"],
                    actual_segment,
                    actual_crop,
                )
            )
        if not isinstance(clip, torch.Tensor) or clip.ndim != 4 or clip.shape[0] != 3:
            raise RuntimeError(
                "native VideoMAE test preprocessing must return a [3,T,H,W] Tensor"
            )
        clip = clip.detach().cpu().contiguous()
        if cache_file is not None:
            self._save_cached_clip(cache_file, clip)
        return clip, entry["video_id"]


@dataclass
class CalibrationSource:
    """Re-iterable unlabeled calibration source plus its replay manifest."""

    manifest: List[Dict[str, Any]]
    loader: DataLoader

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, List[str]]]:
        return iter(self.loader)


def build_calibration_source(
    args,
    *,
    size: int,
    seed: int,
    batch_size: int,
    workers: int,
    cache_dir=None,
) -> CalibrationSource:
    """Select train videos and replay one deterministic native test-mode view each.

    The source annotation labels are never read. A temporary annotation containing
    only train paths and dummy zero labels lets VideoMAE's existing test-mode
    preprocessing run without exposing training targets to the PTQ loader.
    """
    if args.data_set not in DATASET_CLASSES:
        raise NotImplementedError(
            "video PTQ data supports UCF101, Kinetics-400, and SSV2; got {!r}".format(
                args.data_set
            )
        )
    if int(args.nb_classes) != DATASET_CLASSES[args.data_set]:
        raise ValueError(
            "{} expects {} classes, got {}".format(
                args.data_set, DATASET_CLASSES[args.data_set], args.nb_classes
            )
        )
    if int(args.test_num_segment) <= 1 or int(args.test_num_crop) <= 1:
        raise ValueError(
            "native VideoMAE test preprocessing requires test_num_segment and test_num_crop > 1"
        )
    if int(size) <= 0:
        raise ValueError("calibration size must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(workers) < 0:
        raise ValueError("workers must be nonnegative")

    data_root = Path(args.data_path)
    train_annotation = data_root / "train.csv"
    test_annotation = data_root / "test.csv"
    train_paths = _read_path_column(train_annotation)
    test_paths = _read_path_column(test_annotation, allow_empty=True)
    _assert_unique_paths(train_paths, train_annotation)
    train_identities = {_path_identity(path) for path in train_paths}
    overlap = train_identities.intersection(_path_identity(path) for path in test_paths)
    if overlap:
        example = sorted(overlap)[0]
        raise ValueError("train.csv and test.csv contain overlapping video paths: {}".format(example))
    if int(size) > len(train_paths):
        raise ValueError(
            "calibration size {} exceeds {} train videos".format(size, len(train_paths))
        )

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(train_paths))
    selected_indices = permutation[: int(size)]
    view_count = int(args.test_num_segment) * int(args.test_num_crop)
    manifest = []
    for source_index in selected_indices:
        path = train_paths[int(source_index)]
        selected_view = int(rng.integers(view_count))
        segment, crop = divmod(selected_view, int(args.test_num_crop))
        manifest.append(
            {
                "video_id": path,
                "path": path,
                "segment": int(segment),
                "crop": int(crop),
            }
        )

    cache_path = Path(cache_dir) if cache_dir is not None else None
    selected_paths = [entry["path"] for entry in manifest]
    temporary_annotation = _write_dummy_annotation(selected_paths, cache_path)
    try:
        native_dataset = _make_native_test_dataset(args, temporary_annotation)
    finally:
        Path(temporary_annotation).unlink(missing_ok=True)
    _make_decode_fail_fast(native_dataset)
    selected_dataset = _SelectedTestViews(
        native_dataset=native_dataset,
        manifest=manifest,
        train_paths=selected_paths,
        args=args,
        cache_dir=cache_path,
    )
    loader = DataLoader(
        selected_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        drop_last=False,
        collate_fn=_collate_calibration_batch,
    )
    return CalibrationSource(manifest=manifest, loader=loader)


def build_evaluation_loader(args, *, batch_size: int, workers: int, rank=0, world_size=1):
    """Build the unchanged VideoMAE test dataset and its normal evaluation sampler."""
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(workers) < 0:
        raise ValueError("workers must be nonnegative")
    if int(world_size) <= 0 or int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError("rank must be in [0, world_size) with world_size > 0")

    dataset, nb_classes = build_dataset(is_train=False, test_mode=True, args=args)
    if int(nb_classes) != int(args.nb_classes):
        raise ValueError(
            "VideoMAE builder returned {} classes, args.nb_classes is {}".format(
                nb_classes, args.nb_classes
            )
        )
    if int(world_size) > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=False,
        )
    else:
        sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=int(batch_size),
        num_workers=int(workers),
        drop_last=False,
    )
    # Keep the class count discoverable without changing the builder's return API.
    loader.nb_classes = int(nb_classes)
    return loader
