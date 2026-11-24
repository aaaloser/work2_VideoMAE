# VideoMAE ViT-B 浮点基线评估

使用微调后的原始权重和 VideoMAE 的 `--eval --dist_eval` 入口，沿用 UCF101 基线的 CUDA autocast 和视频级多视角汇总。输入为 224×224，每个视角 16 帧。

| 数据集 | 验证视频 | 时间片段 × 空间裁剪 | 视角总数 |
| --- | ---: | ---: | ---: |
| Kinetics-400 | 19,760 | 5 × 3 | 296,400 |
| Something-Something V2 | 24,777 | 2 × 3 | 148,662 |

Kinetics-400 使用采样步长 4，按官方验证标注匹配本地 MP4、MKV 和 WebM 文件。本地可评估 19,760 / 19,906 条验证标注。24 个样本的检查中，按类名排序的标签命中 18 条，`val_model_label.csv` 命中 0 条，因此使用按类名排序的标签。SSv2 使用原始 WebM 和官方验证 JSON，标签与 `val_videofolder.txt` 一致。两份权重均通过严格加载。

## 准备评估标注

生成并核对 Kinetics-400 和 SSv2 的 VideoMAE 视频标注。

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 /data/liyifan24/envs/videomae/bin/python -u - \
  > /data/liyifan24/work2/VideoMAE-main/out_log/videomae_k400_ssv2_prepare.log 2>&1 <<'PY'
import csv
import json
from pathlib import Path

root = Path("/data/liyifan24/work2/VideoMAE-main/annotations")
data = Path("/data/liyifan24/Datasets")
kinetics = data / "Kinetics-400"
something = data / "somethingv2"
webm = something / "prime" / "20bn-something-something-v2"

def read_pairs(path):
    pairs = {}
    for line in path.read_text().splitlines():
        if line.strip():
            name, label = line.rsplit(None, 1)
            if name in pairs:
                raise ValueError(f"Duplicate annotation in {path}: {name}")
            pairs[name] = int(label)
    return pairs

def check_files(rows, expected_classes):
    labels = set()
    ids = set()
    for path, label in rows:
        video_id = path.name.split(".")[0]
        if video_id in ids:
            raise ValueError(f"Duplicate evaluation ID: {path}")
        ids.add(video_id)
        labels.add(label)
        if not path.is_file() or path.stat().st_size < 1024:
            raise ValueError(f"Missing or incomplete video: {path}")
    if labels != set(range(expected_classes)):
        raise ValueError(f"Unexpected evaluation labels: {sorted(labels)}")

def write_lists(directory, train, val):
    destination = root / directory
    destination.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("val", val), ("test", val)):
        (destination / f"{split}.csv").write_text(
            "".join(f"{path} {label}\n" for path, label in rows)
        )
        print(f"{directory} {split}: {len(rows)} videos")

class_names = (kinetics / "kinetics400_classes.txt").read_text().splitlines()
class_index = {name: label for label, name in enumerate(class_names)}
categories = {name.replace(" ", "_"): label for name, label in class_index.items()}
if len(categories) != 400:
    raise ValueError("Unexpected Kinetics-400 class list")
with (kinetics / "label" / "kinetics-400_val.csv").open() as source:
    official_val = {(row["label"].replace(" ", "_"), row["youtube_id"]): class_index[row["label"]]
                    for row in csv.DictReader(source)}
k_val = []
for folder in sorted((kinetics / "val_256").iterdir()):
    if not folder.is_dir() or folder.name not in categories:
        continue
    for path in sorted(folder.iterdir()):
        label = official_val.get((folder.name, path.name.split(".")[0]))
        if label is not None and path.suffix.lower() in {".mp4", ".mkv", ".webm"}:
            k_val.append((path, label))
matched = {str(path.relative_to(kinetics)): label for path, label in k_val}
standard_val = read_pairs(kinetics / "val.csv")
if any(matched.get(path) != label for path, label in standard_val.items()):
    raise ValueError("Official Kinetics-400 labels disagree with local MP4 annotations")
print(f"Kinetics-400 local labeled validation subset: {len(k_val)}/{len(official_val)} annotations")
k_train = [(kinetics / path, label) for path, label in read_pairs(kinetics / "train.csv").items()]
check_files(k_val, 400)
write_lists("kinetics400_val", k_train, k_val)

labels = json.loads((something / "data" / "labels" / "labels.json").read_text())
def ssv2_split(filename):
    records = json.loads((something / "data" / "labels" / filename).read_text())
    return [(webm / f"{record['id']}.webm",
             int(labels[record["template"].replace("[", "").replace("]", "")]))
            for record in records]

s_train = ssv2_split("train.json")
s_val = ssv2_split("validation.json")
if len(s_val) != 24777:
    raise ValueError(f"Unexpected SSV2 validation count: {len(s_val)}")
reference_val = {name: int(label) for name, _, label in
                 (line.split() for line in (something / "val_videofolder.txt").read_text().splitlines())}
if reference_val != {path.stem: label for path, label in s_val}:
    raise ValueError("SSV2 validation JSON and local video-folder labels disagree")
check_files(s_val, 174)
write_lists("ssv2_val", s_train, s_val)
PY
```

## 在 GPU 3 顺序评估

确认 3 号卡空闲后，启动 Kinetics-400 和 SSv2，并在每项结束后核对预测覆盖和视频级准确率。

```bash
nohup setsid env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  bash -s > /data/liyifan24/work2/VideoMAE-main/out_log/videomae_k400_ssv2_float_gpu3_queue.log 2>&1 <<'BASH' &
set -uo pipefail
project=/data/liyifan24/work2/VideoMAE-main
python_bin=/data/liyifan24/envs/videomae/bin/python
checkpoint_dir=/data/liyifan24/VideoMAE-main/checkpoint
run_dir="$project/out_log/videomae_k400_ssv2_float_gpu3"
mkdir -p "$run_dir"
printf '%s\n' "$$" > "$run_dir/queue.pid"

verify_predictions() {
  "$python_bin" -u - "$@" <<'PY'
import json
import sys
from pathlib import Path

annotation, output = map(Path, sys.argv[1:3])
segments = int(sys.argv[3])
expected = {}
for line in annotation.read_text().splitlines():
    path, label = line.split()
    expected[Path(path).name.split(".")[0]] = int(label)

views = {}
rows = 0
with (output / "0.txt").open() as predictions:
    next(predictions)
    for line in predictions:
        video = line.split(" ", 1)[0]
        label, segment, crop = map(int, line.rsplit("]", 1)[1].split())
        if expected.get(video) != label or not (0 <= segment < segments and 0 <= crop < 3):
            raise ValueError(f"Unexpected prediction for {video}")
        views.setdefault(video, set()).add((segment, crop))
        rows += 1
expected_rows = len(expected) * segments * 3
if rows != expected_rows or views.keys() != expected.keys() or any(
    len(grid) != segments * 3 for grid in views.values()
):
    raise ValueError(f"Incomplete evaluation: {len(views)} videos, {rows} rows")

metrics = json.loads((output / "log.txt").read_text().splitlines()[-1])
metrics.update(videos=len(expected), views=rows)
(output / "verified_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics))
PY
}

evaluate() {
  local dataset=$1 slug=$2 classes=$3 segments=$4 ckpt=$5
  local annotations="$project/annotations/${slug}_val"
  local output="$run_dir/$slug"
  printf '[%s] START %s\n' "$(date -Is)" "$dataset"
  nohup "$python_bin" -u -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$project/run_class_finetuning.py" \
    --eval --dist_eval --no_auto_resume --no_save_ckpt \
    --model vit_base_patch16_224 --data_set "$dataset" --nb_classes "$classes" \
    --data_path "$annotations" --finetune "$checkpoint_dir/$ckpt" \
    --model_key module --num_frames 16 --tubelet_size 2 --sampling_rate 4 \
    --input_size 224 --short_side_size 224 --test_num_segment "$segments" --test_num_crop 3 \
    --batch_size 8 --num_workers 8 --output_dir "$output" \
    > "$run_dir/$slug.log" 2>&1 < /dev/null
  local status=$?
  if [ "$status" -eq 0 ]; then
    verify_predictions "$annotations/test.csv" "$output" "$segments"
    status=$?
  fi
  printf '[%s] END %s exit=%s\n' "$(date -Is)" "$dataset" "$status"
  return "$status"
}

evaluate Kinetics-400 kinetics400 400 5 VideoMAE_ViT-B_16x1_224_K400.pth
k400_status=$?
evaluate SSV2 ssv2 174 2 VideoMAE_ViT-B_16x1_224_SSv2.pth
ssv2_status=$?
printf '[%s] DONE Kinetics-400=%s SSV2=%s\n' "$(date -Is)" "$k400_status" "$ssv2_status"
[ "$k400_status" -eq 0 ] && [ "$ssv2_status" -eq 0 ]
BASH
```

队列日志在 `out_log/videomae_k400_ssv2_float_gpu3_queue.log`。两项评估分别写入 `out_log/videomae_k400_ssv2_float_gpu3/kinetics400.log` 和 `out_log/videomae_k400_ssv2_float_gpu3/ssv2.log`，对应子目录的 `verified_metrics.json` 保存通过覆盖核对的结果。此流程只评估已有权重，训练生成的模型使用 `out_checkpoint/`。

## 浮点基线实测结果

Kinetics-400 与 SSv2 均于 2026 年 11 月 23 日在 GPU 3 顺序完成，退出码均为 0。下表指标已从保存的预测逐行复算，与三次评估记录一致。核对包括每个视频的标签、完整视角和有限 logits。 [队列日志](../out_log/videomae_k400_ssv2_float_gpu3_queue.log) · [复核日志](../out_log/videomae_float_baselines_audit.log)

| 数据集 | 视频数 | 唯一视角数 | Top-1 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| UCF101 split 1 | 3,783 | 56,745 | 91.25% | 98.55% |
| Kinetics-400 本地验证集 | 19,760 | 296,400 | 81.26% | 94.91% |
| Something-Something V2 验证集 | 24,777 | 148,662 | 70.53% | 92.42% |

以项目 Model Zoo 的相同视角配置为参考，Kinetics-400 和 SSv2 的 Top-1 分别低 0.24 和 0.27 个百分点。Kinetics-400 本地测试覆盖 19,760 / 19,906 条验证标注。 [VideoMAE Model Zoo](../MODEL_ZOO.md)

### 复算视频级准确率

从三个数据集的预测日志检查标签、视角和 logits，并重新计算视频级 Top-1 和 Top-5。

```bash
nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /data/liyifan24/envs/videomae/bin/python -u - \
  > /data/liyifan24/work2/VideoMAE-main/out_log/videomae_float_baselines_audit.log 2>&1 <<'PY'
import json
from pathlib import Path

import numpy as np

project = Path("/data/liyifan24/work2/VideoMAE-main")
log = project / "out_log"
queue = log / "videomae_k400_ssv2_float_gpu3"
runs = (
    ("UCF101", "ucf101_split1", log / "videomae_vitb_ucf101_split1_5x3", 101, 5, 2, 3783),
    ("Kinetics-400", "kinetics400_val", queue / "kinetics400", 400, 5, 1, 19760),
    ("SSv2", "ssv2_val", queue / "ssv2", 174, 2, 1, 24777),
)
for name, subset, output, classes, segments, ranks, videos in runs:
    annotations = project / "annotations" / subset / "test.csv"
    index = {}
    labels = []
    for entry in annotations.read_text().splitlines():
        path, label = entry.split()
        video_id = Path(path).name.split(".")[0]
        if video_id in index or not 0 <= int(label) < classes:
            raise ValueError(f"{name}: invalid annotation {entry}")
        index[video_id] = len(labels)
        labels.append(int(label))
    if len(index) != videos or len(set(labels)) != classes:
        raise ValueError(f"{name}: malformed split")

    masks = [0] * videos
    sums = np.zeros((videos, classes), dtype=np.float64)
    repeats = 0
    for rank in range(ranks):
        with (output / f"{rank}.txt").open() as predictions:
            next(predictions)
            for row in predictions:
                video_id, body = row.split(" [", 1)
                scores, fields = body.split("]", 1)
                label, segment, crop = map(int, fields.split())
                position = index[video_id]
                if label != labels[position] or not (0 <= segment < segments and 0 <= crop < 3):
                    raise ValueError(f"{name}: invalid label or view for {video_id}")
                logits = np.fromstring(scores, sep=",", dtype=np.float64)
                if logits.shape != (classes,) or not np.isfinite(logits).all():
                    raise ValueError(f"{name}: invalid logits for {video_id}")
                view = 1 << (segment * 3 + crop)
                if masks[position] & view:
                    repeats += 1
                    continue
                masks[position] |= view
                probs = np.exp(logits - logits.max())
                sums[position] += probs / probs.sum()

    views = videos * segments * 3
    if any(mask != (1 << (segments * 3)) - 1 for mask in masks) or repeats != (-views) % ranks:
        raise ValueError(f"{name}: unexpected view coverage or duplicate count")
    targets = np.asarray(labels)
    means = sums / (segments * 3)
    top1_count = int((means.argmax(axis=1) == targets).sum())
    top5_count = int((np.argsort(-means, axis=1)[:, :5] == targets[:, None]).any(axis=1).sum())
    top1 = 100 * top1_count / videos
    top5 = 100 * top5_count / videos
    recorded = json.loads((output / "log.txt").read_text().splitlines()[-1])
    checks = (
        np.isclose(top1, recorded["Final top-1"], rtol=0, atol=1e-10),
        np.isclose(top5, recorded["Final Top-5"], rtol=0, atol=1e-10),
    )
    if not all(checks):
        raise ValueError(f"{name}: recomputed metrics differ from {recorded}")
    verified = output / "verified_metrics.json"
    if verified.is_file():
        report = json.loads(verified.read_text())
        expected = dict(recorded, videos=videos, views=views)
        if report != expected:
            raise ValueError(f"{name}: queue verification report mismatch")
    print(f"{name}: {videos} videos, {views} unique views, {repeats} distributed padding rows; "
          f"Top-1 {top1:.8f}% ({top1_count}/{videos}), Top-5 {top5:.8f}% ({top5_count}/{videos})")
PY
```

