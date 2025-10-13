# GAPNet: A Lightweight Framework for Image and Video Salient Object Detection via Granularity-Aware Paradigm

This repository contains the official PyTorch implementation of GAPNet for image and video salient object detection (SOD/VSOD). GAPNet adopts a granularity-aware paradigm that supervises multi-scale decoder side-outputs with matching label granularities and employs efficient GPC/CSA modules plus a lightweight global self-attention head.

The paper has been accepted by Machine Intelligence Research (MIR) 2025.

Core ideas:
- Granularity-aware deep supervision: high-level side-output supervised by center saliency (low granularity), low-level side-output supervised by boundary+others (high granularity), final output supervised by full saliency.
- Granular Pyramid Convolution (GPC) for efficient low-scale feature fusion.
- Cross-Scale Attention (CSA) for high-level feature fusion and a lightweight global self-attention head over 1/32 features.
- Unified image + video: two-stream RGB+flow variant for VSOD with hierarchical fusion.

Paper: see [25MIR_GAPNet.pdf](https://yuhuan-wu.com/files/[25MIR]%20GAPNet.pdf).



## Requirements

- Python 3.8+
- PyTorch and TorchVision (CUDA recommended)
- OpenCV
- tqdm
- Optional (for FLOPs and speed reporting during test): fvcore

Install via pip:

```bash
pip install -r requirements.txt
# for FLOPs/speed reports used in [scripts/test.py](scripts/test.py) and [scripts/test_video.py](scripts/test_video.py):
pip install fvcore
```


## Repository structure

- Models
  - [models/model.py](models/model.py): GACNet for SOD with MobileNetV2 backbone and transformer-based fusion.
  - [models/model_video.py](models/model_video.py): Two-stream GAPNet for VSOD (RGB + optical flow).
- Training and testing scripts
  - SOD: [scripts/train.sh](scripts/train.sh), [scripts/train.py](scripts/train.py), [scripts/test.sh](scripts/test.sh), [scripts/test.py](scripts/test.py)
  - VSOD: [scripts/train_video.sh](scripts/train_video.sh), [scripts/train_video.py](scripts/train_video.py), [scripts/test_video.sh](scripts/test_video.sh), [scripts/test_video.py](scripts/test_video.py)
- Utilities: [transforms.py](transforms.py), [saleval.py](saleval.py)
- Pretrained: put checkpoints under ./pretrained/ (e.g., [pretrained/gapnet.pth](pretrained/gapnet.pth))


## Pretrained models

[[gapnet.pth]](https://github.com/yuhuan-wu/GAPNet/releases/download/pretrained/gapnet.pth)
[[gapnet_video.pth]](https://github.com/yuhuan-wu/GAPNet/releases/download/pretrained/gapnet_video.pth)
[[imagenet pretrained mobilenetv2]](https://github.com/yuhuan-wu/GAPNet/releases/download/pretrained/mobilenet_v2-b0353104.pth)

[(can manually download them from the github release page)](https://github.com/yuhuan-wu/GAPNet/releases/tag/pretrained)

Place weights under ./pretrained/:

- gapnet*.pth: trained checkpoints for SOD/VSOD. Use them directly with the provided test scripts:
  - SOD: set --pretrained to a SOD checkpoint in [scripts/test.sh](scripts/test.sh)
  - Video SOD: set --pretrained to a video checkpoint in [scripts/test_video.sh](scripts/test_video.sh)
  Examples: [pretrained/gapnet.pth](pretrained/gapnet.pth), [pretrained/gapnet_video.pth](pretrained/gapnet_video.pth)

- mobilenet*.pth: ImageNet-pretrained MobileNetV2 backbone for initializing SOD training. The code auto-loads it when arch=mobilenetv2 and pretrained=True; expected filename is [pretrained/mobilenet_v2-b0353104.pth](pretrained/mobilenet_v2-b0353104.pth). See [models/MobileNetV2.mobilenetv2()](models/MobileNetV2.py:109) for the expected path logic.

Note: VGG pretrained weights are not used in this release.
## Prepare the data

Download and unzip:
- SOD datasets: [HF: SOD_datasets.zip](https://huggingface.co/datasets/yuhuan-wu/GAPNet_datasets/resolve/main/SOD_datasets.zip)
- VSOD datasets: [HF: VSOD_datasets.zip](https://huggingface.co/datasets/yuhuan-wu/GAPNet_datasets/resolve/main/VSOD_datasets.zip)

Place them as:

```text
gacnet-release/
├─ data/                      # default training root for both SOD and VSOD
│  ├─ DUTS-TR/ ...           # images + masks for SOD training
│  ├─ DAVSOD/ ...# frames, flow, masks for VSOD training
│  ├─ DUTS-TE.txt
│  ├─ DUT-OMRON.txt
│  ├─ HKU-IS.txt
│  ├─ ECSSD.txt
│  └─ PASCAL-S.txt
   ├─ DAVIS_test.lst
   ├─ DAVSOD_test.lst
   ├─ FBMS_test.lst
   ├─ ViSal_test.lst
   ├─ SegTrack-V2_test.lst
   └─ ...
```

Notes on list files:
- SOD .txt format expected by [scripts/test.py](scripts/test.py): each line "relative/path/to/image relative/path/to/label" (relative to --data_dir).
- VSOD .lst format expected by [scripts/test_video.py](scripts/test_video.py): each line "relative/RGB/path relative/Flow/path relative/GT/path".

If you already have the datasets organized differently, you can generate the .txt/.lst files accordingly to match your layout.


## How to train: SOD

Quick start with defaults (MobileNetV2 backbone, poly LR, multi-scale):

```bash
# edit GPU ids and savedir if needed in [scripts/train.sh](scripts/train.sh)
bash scripts/train.sh
```

Important args in [scripts/train.sh](scripts/train.sh):
- --data_dir ./data/ by default (set in [scripts/train.py](scripts/train.py) default). Put SOD training data under ./data/.
- --arch mobilenetv2
- --max_epochs 30, --batch_size 32, --num_workers 6
- --lr_mode poly --lr 1.7e-4
- --ms 1 (multi-scale training), --bcedice 1 (BCE+Dice loss), --adam_beta2 0.99
- Checkpoints saved to --savedir, e.g., ./gapnet/seed2023-.../model_XX.pth


## How to train: VSOD (two-stream)

```bash
# ensure data_dir points to video train data and optical flow
# update pretrained backbone path if needed (see --pretrained_model in [scripts/train_video.py](scripts/train_video.py))
bash scripts/train_video.sh
```

Key differences vs SOD:
- Two-stream input (RGB + flow) via [models/model_video.py](models/model_video.py).
- Training list uses split "DAVSOD_DAVIS_train" defined inside the loader (see Dataset in codebase).
- Default grouped LR for backbone can be toggled with --group_lr.


## How to test: SOD

Edit [scripts/test.sh](scripts/test.sh) to point --pretrained to your checkpoint, e.g.:

```bash
PYTHONPATH=$(pwd):$PYTHONPATH  python3 scripts/test.py \
  --pretrained ./pretrained/model_16.pth \
  --arch mobilenetv2 \
  --data_dir ./data-sod \
  --savedir ./outputs
```

This will:
- Report FLOPs and FPS (requires fvcore).
- Save per-dataset predictions under ./outputs/<DatasetName>/.


## How to test: VSOD

Edit [scripts/test_video.sh](scripts/test_video.sh):
- Set MODEL_PATH to your checkpoint.
- Optionally adjust SAVE_DIR, METHOD_TAG, and the datasets array.

Then run:

```bash
bash scripts/test_video.sh
```

Predictions will be saved under SAVE_DIR/METHOD_TAG/<DatasetName>/..., following the path logic inside [scripts/test_video.py](scripts/test_video.py).


## Reproducing paper settings

Training uses:
- Optimizer Adam with betas (0.9, 0.99), weight_decay 1e-4
- Polynomial LR schedule (power 0.9) with warmup
- Input sizes for augmentation: 320/352/384; test size 384
- Diverse supervision and global guidance implemented in the model and loss

You can modify these via CLI flags in [scripts/train.py](scripts/train.py) and [scripts/train_video.py](scripts/train_video.py).


## Citation

If you find this work useful, please cite:

```bibtex
@article{wu2025gapnet,
  title     = {GAPNet: A Lightweight Framework for Image and Video Salient Object Detection via Granularity-Aware Paradigm},
  author    = {Yu-Huan Wu and Wei Liu and Shi-Chen Zhang and Zizhou Wang and Yong Liu and Liangli Zhen},
  journal   = {Machine Intelligence Research},
  year      = {2025},
}
```


## Contact

For questions or collaborations, please contact: wu_yuhuan@a-star.edu.sg


## License

This code is released for academic research. For commercial use, please contact the authors.
