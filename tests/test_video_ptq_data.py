import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import video_ptq.data as video_data


def _args(root, *, data_set="UCF101", segments=3, crops=2):
    classes = {"UCF101": 101, "Kinetics-400": 400, "SSV2": 174}
    return SimpleNamespace(
        data_set=data_set,
        data_path=str(root),
        nb_classes=classes[data_set],
        num_frames=2,
        sampling_rate=3,
        input_size=4,
        short_side_size=4,
        test_num_segment=segments,
        test_num_crop=crops,
    )


def _write_annotations(root, train, test=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.csv").write_text("".join(f"{path} {label}\n" for path, label in train))
    (root / "test.csv").write_text(
        "".join(f"{path} {label}\n" for path, label in (test or []))
    )


def _make_video_files(root, paths):
    for path in paths:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"x" * 1024)


class _FakeNativeTestDataset:
    """Small stand-in for VideoClsDataset's existing deterministic test path."""

    def __init__(self, anno_path, args, *, bad_identity=False, bad_view=False):
        self.args = args
        self.dataset_samples = [
            line.rsplit(maxsplit=1)[0]
            for line in Path(anno_path).read_text().splitlines()
        ]
        self.label_array = [
            int(line.rsplit(maxsplit=1)[1])
            for line in Path(anno_path).read_text().splitlines()
        ]
        self.test_num_segment = args.test_num_segment
        self.test_num_crop = args.test_num_crop
        self.test_dataset = []
        self.test_seg = []
        for segment in range(self.test_num_segment):
            for crop in range(self.test_num_crop):
                for path in self.dataset_samples:
                    self.test_dataset.append(path)
                    self.test_seg.append((segment, crop))
        self.bad_identity = bad_identity
        self.bad_view = bad_view
        self.read_count = 0

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, index):
        self.read_count += 1
        path = self.test_dataset[index]
        segment, crop = self.test_seg[index]
        if self.bad_identity:
            path = str(Path(path).with_name("unexpected.avi"))
        if self.bad_view:
            segment = (segment + 1) % self.test_num_segment
        clip = torch.full((3, self.args.num_frames, 4, 4), float(index))
        # The native dataset returns a label; the PTQ-facing wrapper must drop it.
        return clip, 987, Path(path).name.rsplit(".", 1)[0], segment, crop


def _install_fake_native(monkeypatch, *, bad_identity=False, bad_view=False):
    made = []

    def factory(args, anno_path):
        dataset = _FakeNativeTestDataset(
            anno_path, args, bad_identity=bad_identity, bad_view=bad_view
        )
        made.append(dataset)
        return dataset

    monkeypatch.setattr(video_data, "_make_native_test_dataset", factory)
    return made


def test_calibration_source_uses_seeded_train_prefix_and_returns_unlabeled_batches(
    tmp_path, monkeypatch
):
    train = [(f"{tmp_path}/video_{i}.avi", 50 + i) for i in range(8)]
    test = [(f"{tmp_path}/heldout_{i}.avi", i) for i in range(2)]
    _write_annotations(tmp_path, train, test)
    _make_video_files(tmp_path, [path for path, _ in train])
    made = _install_fake_native(monkeypatch)
    args = _args(tmp_path)

    source = video_data.build_calibration_source(
        args, size=5, seed=7, batch_size=2, workers=0
    )
    repeat = video_data.build_calibration_source(
        args, size=5, seed=7, batch_size=2, workers=0
    )

    assert source.manifest == repeat.manifest
    assert len(source.manifest) == 5
    assert len({entry["video_id"] for entry in source.manifest}) == 5
    assert all(entry["video_id"] in {path for path, _ in train} for entry in source.manifest)
    assert all(0 <= item["segment"] < args.test_num_segment for item in source.manifest)
    assert all(0 <= item["crop"] < args.test_num_crop for item in source.manifest)
    json.dumps(source.manifest)
    assert made[0].dataset_samples == [item["path"] for item in source.manifest]
    assert made[0].label_array == [0] * len(source.manifest)

    batches = list(source.loader)
    assert [len(batch) for batch in batches] == [2, 2, 2]
    for clips, ids in batches:
        assert clips.ndim == 5
        assert clips.shape[1:] == (3, args.num_frames, 4, 4)
        assert isinstance(ids, list)
        assert all(isinstance(video_id, str) for video_id in ids)
    assert sum(len(ids) for _, ids in batches) == 5


@pytest.mark.parametrize("bad_identity,bad_view", [(True, False), (False, True)])
def test_calibration_fails_when_native_test_dataset_falls_back_to_another_identity_or_view(
    tmp_path, monkeypatch, bad_identity, bad_view
):
    train = [(f"{tmp_path}/video_{i}.avi", i) for i in range(3)]
    _write_annotations(tmp_path, train)
    _make_video_files(tmp_path, [path for path, _ in train])
    _install_fake_native(
        monkeypatch, bad_identity=bad_identity, bad_view=bad_view
    )

    source = video_data.build_calibration_source(
        _args(tmp_path), size=2, seed=1, batch_size=1, workers=0
    )
    with pytest.raises(RuntimeError, match="identity|view"):
        list(source.loader)


def test_calibration_rejects_train_test_overlap_using_only_path_columns(tmp_path, monkeypatch):
    shared = f"{tmp_path}/shared.avi"
    _write_annotations(tmp_path, [(shared, 4)], [(shared, 91)])

    with pytest.raises(ValueError, match="overlap"):
        video_data.build_calibration_source(
            _args(tmp_path), size=1, seed=0, batch_size=1, workers=0
        )


def test_optional_cache_prevents_redecoding_on_replay(tmp_path, monkeypatch):
    train = [(f"{tmp_path}/video_{i}.avi", i) for i in range(4)]
    _write_annotations(tmp_path, train)
    _make_video_files(tmp_path, [path for path, _ in train])
    made = _install_fake_native(monkeypatch)

    source = video_data.build_calibration_source(
        _args(tmp_path),
        size=3,
        seed=3,
        batch_size=2,
        workers=0,
        cache_dir=tmp_path / "clip-cache",
    )
    first = [(clips.clone(), ids) for clips, ids in source.loader]
    reads_after_first_pass = made[0].read_count
    second = [(clips.clone(), ids) for clips, ids in source.loader]

    assert reads_after_first_pass == 3
    assert made[0].read_count == reads_after_first_pass
    assert [ids for _, ids in first] == [ids for _, ids in second]
    assert all(torch.equal(a, b) for (a, _), (b, _) in zip(first, second))


def test_evaluation_loader_calls_existing_builder_with_test_arguments_unchanged(
    monkeypatch
):
    args = _args(Path("/unused"))
    dataset = ["view-0", "view-1", "view-2"]
    calls = []

    def builder(is_train, test_mode, args):
        calls.append((is_train, test_mode, args))
        return dataset, 101

    monkeypatch.setattr(video_data, "build_dataset", builder)
    loader = video_data.build_evaluation_loader(
        args, batch_size=2, workers=0, rank=1, world_size=2
    )

    assert calls == [(False, True, args)]
    assert loader.dataset is dataset
    assert loader.nb_classes == 101
    assert isinstance(loader.sampler, torch.utils.data.DistributedSampler)
    assert loader.sampler.rank == 1
    assert loader.sampler.num_replicas == 2
    assert loader.batch_size == 2


@pytest.mark.parametrize(
    "data_set,module_name,class_name,expected_options",
    [
        (
            "UCF101",
            "kinetics",
            "VideoClsDataset",
            {"clip_len": 2, "frame_sample_rate": 3, "num_segment": 1},
        ),
        (
            "Kinetics-400",
            "kinetics",
            "VideoClsDataset",
            {"clip_len": 2, "frame_sample_rate": 3, "num_segment": 1},
        ),
        (
            "SSV2",
            "ssv2",
            "SSVideoClsDataset",
            {"clip_len": 1, "num_segment": 2},
        ),
    ],
)
def test_calibration_uses_each_native_dataset_test_mode_protocol(
    monkeypatch, tmp_path, data_set, module_name, class_name, expected_options
):
    captured = {}

    class RecordingDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType(module_name)
    setattr(fake_module, class_name, RecordingDataset)
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    args = _args(tmp_path, data_set=data_set)

    result = video_data._make_native_test_dataset(args, "train-paths-with-dummy-labels.csv")

    assert isinstance(result, RecordingDataset)
    assert captured["anno_path"] == "train-paths-with-dummy-labels.csv"
    assert captured["mode"] == "test"
    assert captured["data_path"] == "/"
    assert {key: captured[key] for key in expected_options} == expected_options
    assert captured["test_num_segment"] == args.test_num_segment
    assert captured["test_num_crop"] == args.test_num_crop
    assert captured["num_crop"] == 3
    assert captured["args"] is args

