# UCF101
## fp32
```bash
nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /data/liyifan24/envs/videomae/bin/python -u -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  /data/liyifan24/work2/VideoMAE-main/run_class_finetuning.py \
  --eval --dist_eval --no_auto_resume --no_save_ckpt \
  --model vit_base_patch16_224 --data_set UCF101 --nb_classes 101 \
  --data_path /data/liyifan24/work2/VideoMAE-main/annotations/ucf101_split1 \
  --finetune /data/liyifan24/work2/VideoMAE-main/pretrained/VideoMAE_ViT-B_16x1_224_UCF101.pth \
  --model_key module --num_frames 16 --tubelet_size 2 --sampling_rate 4 \
  --input_size 224 --short_side_size 224 --test_num_segment 5 --test_num_crop 3 \
  --batch_size 8 --num_workers 8 \
  --output_dir /data/liyifan24/work2/VideoMAE-main/out_log/videomae_vitb_ucf101_split1_float \
  > /data/liyifan24/work2/VideoMAE-main/out_log/videomae_vitb_ucf101_split1_float.log 2>&1 < /dev/null &
```

## baseline(FIMA-Q)
```bash
nohup /data/liyifan24/envs/videomae/bin/python -u \
  /data/liyifan24/work2/VideoMAE-main/run_video_ptq.py full --device cuda:2 \
  > /data/liyifan24/work2/VideoMAE-main/out_log/ucf101_vitb_fimaq_w4a4_full.log 2>&1 < /dev/null &
```

# Kinetics-400
## fp32
```bash
nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /data/liyifan24/envs/videomae/bin/python -u -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  /data/liyifan24/work2/VideoMAE-main/run_class_finetuning.py \
  --eval --dist_eval --no_auto_resume --no_save_ckpt \
  --model vit_base_patch16_224 --data_set Kinetics-400 --nb_classes 400 \
  --data_path /data/liyifan24/work2/VideoMAE-main/annotations/kinetics400_val \
  --finetune /data/liyifan24/VideoMAE-main/checkpoint/VideoMAE_ViT-B_16x1_224_K400.pth \
  --model_key module --num_frames 16 --tubelet_size 2 --sampling_rate 4 \
  --input_size 224 --short_side_size 224 --test_num_segment 5 --test_num_crop 3 \
  --batch_size 8 --num_workers 8 \
  --output_dir /data/liyifan24/work2/VideoMAE-main/out_log/videomae_vitb_kinetics400_val_float \
  > /data/liyifan24/work2/VideoMAE-main/out_log/videomae_vitb_kinetics400_val_float.log 2>&1 < /dev/null &
```



# Something-Something-v2
## fp32
```bash
nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /data/liyifan24/envs/videomae/bin/python -u -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  /data/liyifan24/work2/VideoMAE-main/run_class_finetuning.py \
  --eval --dist_eval --no_auto_resume --no_save_ckpt \
  --model vit_base_patch16_224 --data_set SSV2 --nb_classes 174 \
  --data_path /data/liyifan24/work2/VideoMAE-main/annotations/ssv2_val \
  --finetune /data/liyifan24/VideoMAE-main/checkpoint/VideoMAE_ViT-B_16x1_224_SSv2.pth \
  --model_key module --num_frames 16 --tubelet_size 2 \
  --input_size 224 --short_side_size 224 --test_num_segment 2 --test_num_crop 3 \
  --batch_size 8 --num_workers 8 \
  --output_dir /data/liyifan24/work2/VideoMAE-main/out_log/videomae_vitb_ssv2_val_float \
  > /data/liyifan24/work2/VideoMAE-main/out_log/videomae_vitb_ssv2_val_float.log 2>&1 < /dev/null &
```
