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
    └── augmentations.py  # Image transformation pipeline, and the different cropping strategies from the paper
    └── ...
```

---

## Key Components

### Dataset Variants

The framework supports multiple fixation-guided cropping strategies, selectable via command-line arguments:

| Dataset Name | Description |
|---|---|
| `dataset_infantFixation` | Crops centered on real infant gaze fixation points |
| `dataset_objectsFixation` | Crops centered on object regions |
| `dataset_centroidFixation` | Crops centered on gaze centroids |
| `dataset_randomFixation` | Random crop baseline |
| `dataset_plainBackground` | Plain background, no fixation bias |

### Model Architecture

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
  -data ./data \
  -dataset_model_train dataset_infantFixation64 \
  -dataset_projection_train dataset_objectsFixation64 \
  -dataset_test dataset_objectsFixation64 \
  --arch resnet18 \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.0005 \
  --temperature 0.07 \
  --out_dim 128
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `-data` | `./data` | Path to the dataset root |
| `-dataset_model_train` | `dataset_infantFixation64` | Fixation strategy for pre-training |
| `-dataset_projection_train` | `dataset_objectsFixation64` | Dataset for linear probe training |
| `-dataset_test` | `dataset_objectsFixation64` | Dataset for linear probe evaluation |
| `--arch` | `resnet18` | Backbone architecture |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `256` | Batch size |
| `--lr` | `0.0005` | Learning rate (AdamW) |
| `--temperature` | `0.07` | Loss temperature |
| `--out_dim` | `128` | Projection head output dimension |
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
  title     = {Toddler-inspired visual object learning},
  author    = {Bambach, Sven and Crandall, David and Smith, Linda and Yu, Chen},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {31},
  year      = {2018}
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
  title     = {Contrastive learning through time},
  author    = {Schneider, Florian and Xu, Xueyi and Ernst, Markus R and Yu, Zhuanghua and Triesch, Jochen},
  booktitle = {SVRHM 2021 Workshop @ NeurIPS},
  year      = {2021},
  month     = {December}
}
```

```
