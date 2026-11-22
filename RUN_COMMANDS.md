# Work2 运行命令

在项目根目录 `/data/liyifan24/work2/VideoMAE-main` 执行。后续按方法、数据集和实验目的追加命令，每条命令前用一句话说明用途。使用 `nohup` 启动实验，日志写入 `out_log/`，训练生成的模型写入 `out_checkpoint/`。

## VideoMAE

### UCF101 split 1

#### 检查权重

检查权重字典、输入投影和 101 类分类头。

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 /data/liyifan24/envs/videomae/bin/python -u - > /data/liyifan24/work2/VideoMAE-main/out_log/ucf101_checkpoint_inspect.log 2>&1 <<'PY'
from collections import Counter
from pathlib import Path
import torch

checkpoint = torch.load(next(Path('pretrained').glob('*.pth')), map_location='cpu')
print('checkpoint keys:', list(checkpoint)[:20])
for key in ('meta', 'epoch', 'args'):
    if key in checkpoint:
        print(key, str(checkpoint[key])[:20000])
state = next((checkpoint[k] for k in ('state_dict', 'model', 'module') if k in checkpoint), checkpoint)
print('parameter groups:', Counter(k.split('.')[0] for k in state))
print('first parameters:', [(k, tuple(v.shape), str(v.dtype)) for k, v in list(state.items())[:50]])
print('selected parameters:', [(k, tuple(v.shape), str(v.dtype)) for k, v in state.items() if 'patch_embed' in k or 'pos_embed' in k or 'fc_norm' in k or 'head' in k])
PY
```

#### 准备标注

将 split 1 标注转换为项目需要的视频路径和类别，并核对样本数与标签。

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 /data/liyifan24/envs/videomae/bin/python -u - > /data/liyifan24/work2/VideoMAE-main/out_log/ucf101_split1_annotations.log 2>&1 <<'PY'
from collections import Counter
from pathlib import Path

project = Path("/data/liyifan24/work2/VideoMAE-main")
dataset = Path("/data/liyifan24/Datasets/UCF101_frame")
video_root = dataset / "video" / "UCF-101"
output = project / "annotations" / "ucf101_split1"
class_dirs = {folder.name.casefold(): folder for folder in video_root.iterdir()
              if folder.is_dir()}
if len(class_dirs) != 101:
    raise ValueError(f"Unexpected class directories: {video_root}")

def read_split(split, expected_count):
    source = dataset / f"ucf101_split1_{split}_rawframes.txt"
    records = {}
    for line in source.read_text().splitlines():
        path, _, label = line.split()
        category, stem = path.split("/")
        video = class_dirs[category.casefold()] / f"{stem}.avi"
        if not video.is_file() or video.stat().st_size < 1024:
            raise ValueError(f"Missing or incomplete video: {video}")
        if video in records:
            raise ValueError(f"Duplicate video: {video}")
        records[video] = int(label)
    if len(records) != expected_count or set(records.values()) != set(range(101)):
        raise ValueError(f"Unexpected split contents: {source}")

    rgb_source = dataset / f"{split}_rgb_split1.txt"
    rgb_labels = Counter((line.split()[0], int(line.split()[2]))
                         for line in rgb_source.read_text().splitlines())
    if rgb_labels != Counter((video.stem, label) for video, label in records.items()):
        raise ValueError(f"Video names or labels disagree with {rgb_source}")
    return records

train = read_split("train", 9537)
test = read_split("val", 3783)
if train.keys() & test.keys():
    raise ValueError("Training and test lists overlap")

output.mkdir(parents=True, exist_ok=True)
for split, records in (("train", train), ("val", test), ("test", test)):
    (output / f"{split}.csv").write_text(
        "".join(f"{video} {label}\n" for video, label in records.items())
    )
    print(f"{split}: {len(records)} videos, {len(set(records.values()))} classes")
PY
```

生成 9,537 条训练标注和 3,783 条测试标注，`val.csv` 与 `test.csv` 使用相同的 split 1 测试列表。

#### CPU 验证

用两个测试视频验证严格加载权重、视频解码、多视角推理和结果汇总。

```bash
nohup env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1 \
  /data/liyifan24/envs/videomae/bin/python -u - \
  > /data/liyifan24/work2/VideoMAE-main/out_log/videomae_vitb_ucf101_split1_cpu_smoke.log 2>&1 <<'PY'
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from timm.models import create_model

from datasets import build_dataset
from engine_for_finetuning import final_test, merge
import modeling_finetune

project = Path("/data/liyifan24/work2/VideoMAE-main")
source = project / "annotations" / "ucf101_split1" / "test.csv"
output = project / "out_log" / "videomae_vitb_ucf101_split1_cpu_smoke"
output.mkdir(parents=True, exist_ok=True)

by_class = {}
for row in source.read_text().splitlines():
    path, label = row.split()
    by_class.setdefault(int(label), (path, label))
selected = [by_class[label] for label in (0, 100)]

model = create_model("vit_base_patch16_224", pretrained=False, num_classes=101,
                     all_frames=16, tubelet_size=2)
weights = project / "pretrained" / "VideoMAE_ViT-B_16x1_224_UCF101.pth"
model.load_state_dict(torch.load(weights, map_location="cpu")["module"], strict=True)
print(f"Strict checkpoint load passed: {weights.name}")
print("Selected test videos:", selected)

with TemporaryDirectory(dir=output) as annotation_dir:
    (Path(annotation_dir) / "test.csv").write_text(
        "".join(f"{path} {label}\n" for path, label in selected)
    )
    args = SimpleNamespace(data_set="UCF101", data_path=annotation_dir, nb_classes=101,
                           num_frames=16, sampling_rate=4, input_size=224,
                           short_side_size=224, test_num_segment=5, test_num_crop=3)
    dataset, _ = build_dataset(is_train=False, test_mode=True, args=args)

loader = torch.utils.data.DataLoader(dataset, batch_size=2)
predictions = output / "0.txt"
final_test(loader, model, torch.device("cpu"), predictions)
rows = predictions.read_text().splitlines()[1:]
counts = Counter(row.split(" [", 1)[0] for row in rows)
expected = Counter({Path(path).stem: 15 for path, _ in selected})
if counts != expected:
    raise ValueError(f"Unexpected prediction counts: {counts}")
top1, top5 = merge(str(output), 1)
print(f"CPU smoke completed: {len(counts)} videos, {len(rows)} views, "
      f"Top-1 {top1:.2f}%, Top-5 {top5:.2f}%")
PY
```


CPU 小样本验证通过。两个视频共生成 30 条视角预测，并成功汇总为视频级指标。

#### 检查测试视频解码

按评估采样步长检查 3,783 个测试视频能否完成解码。

```bash
nohup env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /data/liyifan24/envs/videomae/bin/python -u - \
  > /data/liyifan24/work2/VideoMAE-main/out_log/ucf101_split1_decode.log 2>&1 <<'PY'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decord import VideoReader, cpu

annotations = Path("/data/liyifan24/work2/VideoMAE-main/annotations/ucf101_split1/test.csv")
videos = [line.split()[0] for line in annotations.read_text().splitlines()]

def verify(video):
    reader = VideoReader(video, num_threads=1, ctx=cpu(0))
    frame_ids = list(range(0, len(reader), 4))
    if not frame_ids:
        raise ValueError(f"No frames in {video}")
    reader.seek(0)
    reader.get_batch(frame_ids).asnumpy()

failures = []
with ThreadPoolExecutor(max_workers=6) as pool:
    futures = [pool.submit(verify, video) for video in videos]
    for video, future in zip(videos, futures):
        try:
            future.result()
        except Exception as error:
            failures.append(video)
            print(f"{video}: {error}")

print(f"Decoded {len(videos) - len(failures)}/{len(videos)} test videos")
if failures:
    raise SystemExit(1)
PY
```

3,783 个测试视频全部通过解码检查。

#### 全量 GPU 评估

用 2、3 号卡评估 split 1 的全部 3,783 个测试视频。

```bash
nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /data/liyifan24/envs/videomae/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  /data/liyifan24/work2/VideoMAE-main/run_class_finetuning.py \
  --eval --dist_eval --no_auto_resume --no_save_ckpt \
  --model vit_base_patch16_224 --data_set UCF101 --nb_classes 101 \
  --data_path /data/liyifan24/work2/VideoMAE-main/annotations/ucf101_split1 \
  --finetune /data/liyifan24/work2/VideoMAE-main/pretrained/VideoMAE_ViT-B_16x1_224_UCF101.pth \
  --model_key module --num_frames 16 --tubelet_size 2 --sampling_rate 4 \
  --input_size 224 --short_side_size 224 --test_num_segment 5 --test_num_crop 3 \
  --batch_size 8 --num_workers 8 \
  --output_dir /data/liyifan24/work2/VideoMAE-main/out_log/videomae_vitb_ucf101_split1_5x3 \
  > /data/liyifan24/work2/VideoMAE-main/out_log/videomae_vitb_ucf101_split1_5x3.log 2>&1 < /dev/null &
```

每个视角输入 16 帧，采样步长为 4，使用 5 个时间片段和 3 个空间裁剪。参数依据项目附带的 UCF101 脚本。评估直接加载已有权重，预测与指标写入 `out_log/videomae_vitb_ucf101_split1_5x3/`。

首次启动未留下进程。2、3 号卡随后被占用，共享 GPU 的启动申请被自动审批拒绝。两张卡释放后重新启动评估。

视频级 Top-1 为 **91.25%**，Top-5 为 **98.55%**。评估覆盖 3,783 个视频和 56,745 个不同视角。评估脚本末行把视角总数标作 `test videos`，最终指标按视频汇总。

#### 核对评估输出

核对两张卡的输出是否覆盖所有视频和视角，并读取最终指标。

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 /data/liyifan24/envs/videomae/bin/python -u - \
  > /data/liyifan24/work2/VideoMAE-main/out_log/videomae_vitb_ucf101_split1_5x3_verify.log 2>&1 <<'PY'
from collections import Counter
import json
from pathlib import Path

import numpy as np

project = Path("/data/liyifan24/work2/VideoMAE-main")
annotations = project / "annotations" / "ucf101_split1" / "test.csv"
output = project / "out_log" / "videomae_vitb_ucf101_split1_5x3"
labels = {}
for line in annotations.read_text().splitlines():
    video, label = line.split()
    labels[Path(video).stem] = int(label)

views = Counter()
for rank in (0, 1):
    with (output / f"{rank}.txt").open() as predictions:
        next(predictions)
        for row in predictions:
            video, _, prediction = row.partition(" [")
            scores, _, fields = prediction.partition("]")
            label, segment, crop = map(int, fields.split())
            logits = np.fromstring(scores, dtype=np.float64, sep=",")
            if logits.size != 101 or not np.isfinite(logits).all() or labels[video] != label:
                raise ValueError(f"Unexpected scores or label for {video}")
            views[video, segment, crop] += 1

expected = {(video, segment, crop) for video in labels
            for segment in range(5) for crop in range(3)}
if len(labels) != 3783 or views.keys() != expected:
    raise ValueError("Test video or view coverage mismatch")

padding = sum(views.values()) - len(views)
if padding != 1:
    raise ValueError(f"Unexpected duplicate prediction rows: {padding}")

metrics = json.loads((output / "log.txt").read_text().splitlines()[-1])
top1, top5 = metrics["Final top-1"], metrics["Final Top-5"]
if not 0 <= top1 <= top5 <= 100:
    raise ValueError(f"Invalid metrics: {metrics}")
print(f"Verified {len(labels)} videos and {len(views)} unique views")
print(f"Distributed padding rows: {padding}")
print(f"Top-1: {top1:.4f}%, Top-5: {top5:.4f}%")
PY
```
