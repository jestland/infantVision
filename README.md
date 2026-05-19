# Toddlers’ Active Gaze Behavior Supports Self-Supervised Object Learning

> **Official repository** for the paper accepted at *Developmental Science*.

This project aims to investigate how toddlers’ gaze-driven first-person visual experience contributes to the emergence of robust object representations.

---

## Repository Structure

```
infantVision/
├── main.py               # Entry point: argument parsing, dataset/model setup, training launch
├── simclrbuilder.py      # SimCLR-TT training loop with AMP, TensorBoard logging, and checkpointing
├── loss.py               # Loss function
├── utils.py              # Embedding generation, checkpoint saving, config serialization, accuracy
├── models/
│   └── simclr.py         # backbone and projection head
├── data_aug/
│   └── dataloader.py     # Dataset classes for various fixation strategies
└── tools/
    ├── augmentations.py  # Image transformation pipeline, and the different cropping strategies from the paper
    └── ...
```

---


## Dataset Variants

The framework supports multiple fixation-guided cropping strategies, selectable via command-line arguments:

| Dataset Name | Description |
|---|---|
| `infant_fixation` | Crops centered on recorded toddler gaze locations |
| `random_fixation` | Random crop baseline |
| `center_fixation` | Centroid / no-eye-movement baseline |
| `objects_train` | Labeled object fixation training split for linear probe |
| `objects_test` | Labeled object fixation test split |

## Model Architecture

`ResNetSimCLR` uses a ResNet backbone (default: `resnet18`) with a projection head that outputs 128-dimensional embeddings. During training, both a **representation** and a **projection** are returned; the projection is used for the contrastive loss.


---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/trieschlab/infantVision.git
cd infantVision
 
# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
 
# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py \
  --data ./data/shift \
  --dataset-model-train infant_fixation \
  --dataset-projection-train objects_train \
  --dataset-test objects_test \
  --crop-size 128 \
  --arch resnet18 \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-2 \
  --weight-decay 1e-4 \
  --temperature 0.08 \
  --out-dim 128
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `./data` | Path to the dataset root |
| `--dataset-model-train` | `infant_fixation` | Dataset used for self-supervised pretraining |
| `--dataset-projection-train` | `objects_train` | Labeled object fixation training split for linear probe |
| `--dataset-test` | `objects_test` | Labeled object fixation test split |
| `--crop-size` | `128` | Crop size in pixels |
| `--arch` | `resnet18` | Backbone architecture |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `256` | Batch size |
| `--lr` | `1e-2` | AdamW learning rate |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--temperature` | `0.08` | SimCLR-TT temperature |
| `--out-dim` | `128` | Projection head output dimension |
| `--disable-cuda` | `False` | Force CPU training |


---

## Requirements

- Python 3.8+
- torch>=1.10
- torchvision>=0.11
- numpy>=1.21
- Pillow>=8.0
- tqdm>=4.62
- scikit-learn>=1.0
- tensorboard>=2.8
- matplotlib>=3.5
- PyYAML>=6.0
- opencv-python>=4.5
- pandas>=1.3

---

## Data Availability

The dataset used in this research cannot be made publicly available due to privacy policies. For related work on toddler/adult visual data collection, please refer to:

```bibtex
@article{bambach2018toddler,
  title={Toddler-inspired visual object learning},
  author={Bambach, Sven and Crandall, David and Smith, Linda and Yu, Chen},
  journal={Advances in neural information processing systems},
  volume={31},
  year={2018}
}
```

---

## Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@misc{yu2025toddlersactivegazebehavior,
      title={Toddlers' Active Gaze Behavior Supports Self-Supervised Object Learning}, 
      author={Zhengyang Yu and Arthur Aubret and Marcel C. Raabe and Jane Yang and Chen Yu and Jochen Triesch},
      year={2025},
      eprint={2411.01969},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.01969}, 
}
```

For the contrastive learning through time (CLTT) framework, please refer to:

```bibtex
@inproceedings{schneider2021contrastive,
  title={Contrastive learning through time},
  author={Schneider, Felix and Xu, Xia and Ernst, Markus R and Yu, Zhengyang and Triesch, Jochen},
  booktitle={Svrhm 2021 workshop@ neurips},
  year={2021}
}
```

```
