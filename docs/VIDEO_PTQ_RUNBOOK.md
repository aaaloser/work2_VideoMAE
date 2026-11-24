# VideoMAE FIMA-Q PTQ 运行命令

入口为 `run_video_ptq.py`。主配置 `--stem a4` 覆盖 tubelet 投影、Linear 和分类头的 W4A4，以及两处 Attention MatMul 的 A4×A4。`--stem a8` 将首层投影切换到实际执行 A8 Q/DQ 的 W4A8 对照。FIMA-Q 本地 4-bit 配置把卷积激活位宽设为 8，但 `quant_input()` 在位宽达到 8 时直接返回浮点输入。`--stem reference` 复现该输入旁路。校准和重建使用训练集视频路径，最终评估复用 VideoMAE 的测试 loader、`final_test` 和 `merge`。模型文件是 fake-quant 检查点。 [FIMA-Q 卷积激活路径](../FIMA-Q-main/quant_layers/conv.py#L55)

## UCF101 split 1

确认 GPU 2 空闲后，直接执行下面的命令。GPU 3 空闲时将 `cuda:2` 改为 `cuda:3`。`full` 使用本地 UCF101 split 1 的标注和微调权重，依次完成校准、重建及两份检查点的全量评估。

```bash
nohup /data/liyifan24/envs/videomae/bin/python -u \
  /data/liyifan24/work2/VideoMAE-main/run_video_ptq.py full --device cuda:2 \
  > /data/liyifan24/work2/VideoMAE-main/out_log/ucf101_vitb_fimaq_w4a4_full.log 2>&1 < /dev/null &
```

默认 run 名为 `ucf101_vitb_fimaq_w4a4`。加 `--stem a8` 可运行首层 W4A8 对照，加 `--stem reference` 可运行参考激活旁路。两种对照都有各自的默认 run 名，命令行的日志文件名可随之调整。`--run` 可指定自定义名称。

已有校准检查点时，可以用 `reconstruct` 单独重跑重建。已有目标检查点时，可以用 `evaluate` 单独评估。两条命令使用与 `full` 相同的默认路径和 run 名。

```bash
nohup /data/liyifan24/envs/videomae/bin/python -u \
  /data/liyifan24/work2/VideoMAE-main/run_video_ptq.py reconstruct --device cuda:2 \
  > /data/liyifan24/work2/VideoMAE-main/out_log/ucf101_vitb_fimaq_w4a4_reconstruct.log 2>&1 < /dev/null &

nohup /data/liyifan24/envs/videomae/bin/python -u \
  /data/liyifan24/work2/VideoMAE-main/run_video_ptq.py evaluate --device cuda:2 --checkpoint-stage reconstructed \
  > /data/liyifan24/work2/VideoMAE-main/out_log/ucf101_vitb_fimaq_w4a4_evaluate.log 2>&1 < /dev/null &
```

默认检查点写入 `out_checkpoint/<run>/calibrated.pth` 和 `out_checkpoint/<run>/reconstructed.pth`。视频采样缓存位于该 run 的 `cache/` 目录。评估每卡预测及指标位于 `out_log/<run>/<stage>/`。UCF101 split 1 应包含 3,783 个视频和 56,745 个唯一视角。现有浮点参照为视频级 Top-1 91.2503%、Top-5 98.5461%。

`--forward-batch-size` 控制设备上同时处理的 clip 数，`--candidate-chunk` 控制校准候选并行数。默认两个值都为 1，候选总数仍是 128。重建的逻辑优化 batch 和 Fisher KL batch 分别由 `--recon-batch-size` 和 `--fisher-batch-size` 控制。若显存限制只发生在评估阶段，可以降低 `--eval-batch-size`，保持原 5×3 视角和视频级聚合。临时缓存需要磁盘空间，1,024 个 FP32 clip 约占 9.2 GiB，单次 Attention 算子校准缓存还可能超过 14 GiB。

大张量的校准分位数使用磁盘映射上的精确全量统计。FIMA-Q 参考实现遇到超大数组时会分片求分位数并取均值，这两种统计结果可能不同。候选评分按样本、候选及 Attention query 行分块，保留原误差归约。

如果 GPU 2 和 3 都空闲，可用两张卡评估。此例对重建检查点运行，基础校准检查点对应 `--checkpoint-stage calibrated`。

```bash
nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
  /data/liyifan24/envs/videomae/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  /data/liyifan24/work2/VideoMAE-main/run_video_ptq.py evaluate \
  --checkpoint-stage reconstructed --eval-batch-size 8 \
  > /data/liyifan24/work2/VideoMAE-main/out_log/ucf101_vitb_fimaq_w4a4_eval_reconstructed_2gpu.log 2>&1 < /dev/null &
```

## Kinetics-400 与 SSV2

PTQ 入口复用现有 VideoMAE 的 Dataset 分支。准备与各自浮点基线一致的标注目录、采样参数和微调检查点。`train.csv` 的第一列供校准选择视频，`test.csv` 提供评估的路径和标签。

| 数据集 | 参数 | VideoMAE ViT-B 默认测试视角 | 分类头 |
|---|---|---|---:|
| UCF101 | `--data-set UCF101` | 5×3，16 帧，抽帧步长 4 | 101 |
| Kinetics-400 | `--data-set Kinetics-400` | 5×3，16 帧，抽帧步长 4 | 400 |
| Something-Something V2 | `--data-set SSV2` | 2×3，16 帧，使用原 SSV2 Dataset 的交错帧采样 | 174 |

## CPU 验证

```bash
nohup env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /data/liyifan24/envs/videomae/bin/python -u -m pytest tests/test_video_ptq_*.py -v \
  > /data/liyifan24/work2/VideoMAE-main/out_log/video_ptq_unit.log 2>&1 < /dev/null
```

运行真实 UCF101 视频的 raw 对照。

```bash
nohup /data/liyifan24/envs/videomae/bin/python -u \
  /data/liyifan24/work2/VideoMAE-main/run_video_ptq.py raw-check --device cpu --calib-size 1 --workers 0 \
  > /data/liyifan24/work2/VideoMAE-main/out_log/video_ptq_ucf101_raw_cpu.log 2>&1 < /dev/null
```

本地 CPU raw 对照用真实 UCF101 视频和原 101 类微调检查点，74 个目标算子处于 raw 模式，适配前后 logits 最大绝对误差为 0。 [CPU raw 日志](../out_log/video_ptq_ucf101_raw_cpu.log)

## 本地验证结果

- 指定 VideoMAE 环境下的 30 个 CPU 测试全部通过，覆盖 `full` 阶段顺序、算子、两阶段检查点恢复、DPLR 和三套数据接口。 [测试日志](../out_log/video_ptq_full_cli_tests.log)
- RTX 3090 GPU 2 的真实 UCF101 raw 对照在 CUDA FP32 与 autocast 下均得到零偏差。折叠后的严格 FP32 对照关闭 TF32，最大绝对误差为 `4.77e-7`。 [CUDA raw 日志](../out_log/video_ptq_ucf101_raw_cuda.log) · [折叠诊断](../out_log/video_ptq_ucf101_fold_diagnostic.log)
- GPU 2 上以两个训练片段完成 74 个算子的 W4A4 校准，并以 rank 2、每单元 4 步完成 14 个单元的 Fisher/DPLR 重建。峰值 CUDA 已分配显存分别为 0.906 GiB 和 4.612 GiB。两份检查点都重新加载成功，并对两个留出的测试视频完成各 30 个唯一视角的聚合。本次规模用于验证功能和检查点往返。正式精度与显存结果需按上文完整预算测量。 [校准日志](../out_log/video_ptq_ucf101_calibrate_smoke_cuda_retry.log) · [重建日志](../out_log/video_ptq_ucf101_recon_smoke_cuda.log) · [评估输出](../out_log/ucf101_vitb_fimaq_smoke/)

量化与重建的来源为 Wu 等人的 FIMA-Q。移植保留非对称候选搜索、等价重参数化、AdaRound、QDrop 与 DPLR 损失。 [FIMA-Q 实现](https://github.com/ShiheWang/FIMA-Q) · [FIMA-Q Apache 2.0 许可](FIMA_Q_LICENSE.txt) · [适配方案](FIMA_Q_VIDEOMAE_W4A4_PLAN.md)
