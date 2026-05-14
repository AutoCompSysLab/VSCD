# VSCD: Video-based Scene Change Detection in Unaligned Scenes

This repository provides the official PyTorch implementation of the paper  
**"VSCD: Video-based Scene Change Detection in Unaligned Scenes" (ICML 2026).**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Paper](https://img.shields.io/badge/ICML%202026-Paper-blue)](PAPER_LINK)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/jiae1234/vscd)

<p align="center">
  Jiae Yoon · Ue-Hwan Kim<br>
  <strong>ICML 2026</strong>
</p>

<p align="center">
  <img src="assets/teaser_pre.PNG" width="90%">
</p>

<p align="center">
  <img src="assets/teaser_ours.PNG" width="90%">
</p>

<p align="center">
  VSCD introduces a video-based scene change detection setting with unconstrained
  camera motion, strong cross-view misalignment, and multiple object-level changes.
</p>

## Highlights

- **Video-based Scene Change Detection (VSCD)**: predicts a dense change mask for each query frame given an unaligned reference video and query video.
- **Large-scale VSCD benchmark**: provides reference-query video pairs with query-aligned pixel-wise change masks.
- **Synthetic and real-world evaluation**: includes a large synthetic benchmark and a real-world test set for sim-to-real evaluation.
- **VSCDNet**: a query-centric multi-reference model with frame-level alignment, patch-level correspondence, confidence-aware feature fusion, and query-guided high-resolution decoding.
- **Real-world validation**: demonstrated on a mobile robot for visual surveillance and object incremental learning.

## Method Overview

<p align="center">
  <img src="assets/model_architecture.PNG" width="95%">
</p>

VSCDNet consists of three main stages.

1. **Frame-level alignment**  
   Each reference/query keyframe is encoded with a frozen SAM ViT image encoder. A learnable alignment token and average-pooled patch tokens are used to build frame descriptors. A soft frame-matching distribution is computed between query and reference frames.

2. **Patch-level correspondence**  
   For each query-reference candidate pair, VSCDNet computes local patch correlation within a `k × k` window on the 64×64 token grid and warps reference features toward the query feature grid.

3. **Confidence-aware feature fusion and decoding**  
   Candidate-level change features are fused using frame-level confidence and patch-level confidence. The fused feature is decoded once per query frame into a full-resolution change mask, with query RGB injection at high-resolution decoding stages.

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/AutoCompSysLab/VSCD.git
cd VSCD
```

### 2. Create environment

```bash
conda create -n vscd python=3.10 -y
conda activate vscd
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` is:

```text
torch
torchvision
numpy
pillow
tqdm
opencv-python
```

### 4. Prepare Segment Anything

VSCDNet uses a frozen SAM ViT-B image encoder. Please clone the official Segment Anything repository and download the SAM ViT-B checkpoint.

```bash
git clone https://github.com/facebookresearch/segment-anything.git
```

Expected paths:

```bash
SAM_ROOT=/path/to/segment-anything
SAM_CKPT=/path/to/sam_vit_b_01ec64.pth
```

## Training

Default training follows the paper setting:

- SAM ViT-B frozen image encoder,
- input resolution: 1024×1024,
- token grid: 64×64,
- `T_key = 32`,
- `K = 4`,
- `|S_t|max = 6`,
- local patch window `k = 5`,
- `Lmax = 5`,
- softmax temperature `τ_f = 0.5`,
- BCE-with-logits + soft Dice loss.

Run:

```bash
DATA_ROOT=/path/to/vscd_dataset \
SAM_ROOT=/path/to/segment-anything \
SAM_CKPT=/path/to/sam_vit_b_01ec64.pth \
bash scripts/train_default.sh
```

Or directly:

```bash
python train.py \
  --data_root /path/to/vscd_dataset \
  --sam_root /path/to/segment-anything \
  --sam_ckpt /path/to/sam_vit_b_01ec64.pth \
  --out_dir ./runs/vscd_default \
  --device cuda \
  --epochs 80 \
  --seed 0 \
  --batch_size 2 \
  --num_workers 2 \
  --num_frames_fixed 32 \
  --backbone_chunk 2 \
  --amp \
  --topk_ref_per_t 4 \
  --max_ref_cands_per_t 6 \
  --local_k 5 \
  --max_msp_len 5 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --val_thr 0.5 \
  --softmax_temp 0.5
```

Checkpoints are saved under:

```text
./runs/vscd_default/
  ckpt_current.pt
  ckpt_best.pt
  ckpt_last.pt
```

## Evaluation

Evaluate a trained checkpoint:

```bash
DATA_ROOT=/path/to/vscd_dataset \
SAM_ROOT=/path/to/segment-anything \
SAM_CKPT=/path/to/sam_vit_b_01ec64.pth \
MODEL_CKPT=./runs/vscd_default/ckpt_best.pt \
bash scripts/eval_default.sh
```

Or directly:

```bash
python eval.py \
  --data_root /path/to/vscd_dataset \
  --split test \
  --sam_root /path/to/segment-anything \
  --sam_ckpt /path/to/sam_vit_b_01ec64.pth \
  --model_ckpt ./runs/vscd_default/ckpt_best.pt \
  --num_frames_fixed 32 \
  --batch_size 1 \
  --num_workers 2 \
  --amp \
  --thr 0.5
```

To save predicted masks:

```bash
python eval.py \
  --data_root /path/to/vscd_dataset \
  --split test \
  --sam_root /path/to/segment-anything \
  --sam_ckpt /path/to/sam_vit_b_01ec64.pth \
  --model_ckpt ./runs/vscd_default/ckpt_best.pt \
  --save_dir ./outputs/vscd_default \
  --save_pred
```

## Dataset

The VSCD dataset will be released on Hugging Face:

```text
https://huggingface.co/datasets/jiae1234/vscd
```

The benchmark contains reference-query video pairs from the same indoor environment, captured at different times and along different camera trajectories. For each query video, pixel-wise change masks are provided for object-level changes such as appearances, disappearances, and relocations.

### Dataset Structure

The code expects the dataset to be organized as follows:

```text
<DATA_ROOT>/
  train/
    <space_name>/
      scene0_frames/
        scene0_0000.jpg
        scene0_0001.jpg
        ...
      scene1_frames/
        scene1_0000.jpg
        scene1_0001.jpg
        ...
      scene2_frames/
      scene3_frames/
      scene4_frames/
      change_mask_scene0_to_scene1_frames/
        change_mask_scene0_to_scene1_0000.jpg
        change_mask_scene0_to_scene1_0001.jpg
        ...
      change_mask_scene1_to_scene2_frames/
      change_mask_scene2_to_scene3_frames/
      change_mask_scene3_to_scene4_frames/
      change_mask_scene4_to_scene0_frames/

  test/
    <space_name>/
      scene0_frames/
      scene1_frames/
      scene2_frames/
      scene3_frames/
      scene4_frames/
      change_mask_scene0_to_scene1_frames/
      change_mask_scene1_to_scene2_frames/
      change_mask_scene2_to_scene3_frames/
      change_mask_scene3_to_scene4_frames/
      change_mask_scene4_to_scene0_frames/
```

The default directed scene pairs are:

```text
0 -> 1
1 -> 2
2 -> 3
3 -> 4
4 -> 0
```

Each training/evaluation sample consists of:

- a reference video from `scene{a}_frames/`,
- a query video from `scene{b}_frames/`,
- query-aligned change masks from `change_mask_scene{a}_to_scene{b}_frames/`.

## Citation

If you find this repository or dataset useful, please cite:

```bibtex
@inproceedings{yoon2026vscd,
  title     = {VSCD: Video-based Scene Change Detection in Unaligned Scenes},
  author    = {Yoon, Jiae and Kim, Ue-Hwan},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgements
This implementation is built upon:

* Segment Anything Model (SAM)
  [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

We thank the authors of Segment Anything for releasing their code and pretrained models.

