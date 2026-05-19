import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_transform(crop_size: int):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
    ])


class LabeledDatasets(Dataset):
    def __init__(self, root_dir, transform=None, split="train"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        class_names = sorted(
            d.name for d in split_dir.iterdir() if d.is_dir()
        )

        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            class_dir = split_dir / class_name

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {split_dir}")

        self.n_classes = len(class_names)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class InfantVisionDatasets(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.pairs = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        for folder_path in sorted(self.root_dir.iterdir()):
            if not folder_path.is_dir():
                continue

            imgs = sorted(
                p for p in folder_path.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            )

            for i in range(len(imgs) - 1):
                self.pairs.append((imgs[i], imgs[i + 1], folder_path.name))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No temporal image pairs found in {self.root_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, folder_name = self.pairs[idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, folder_name


def build_dataset(name, data_root, crop_size=128):
    data_root = Path(data_root)
    transform = make_transform(crop_size)
    size_dir = f"{crop_size}x{crop_size}"

    dataset_roots = {
        "infant_fixation": data_root / "fixation cropping" / size_dir,
        "random_fixation": data_root / "random cropping" / size_dir,
        "center_fixation": data_root / "center cropping" / size_dir,
        "objects_train": data_root / "objects fixation" / size_dir,
        "objects_test": data_root / "objects fixation" / size_dir,
    }

    if name not in dataset_roots:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available datasets: {list(dataset_roots.keys())}"
        )

    root_dir = dataset_roots[name]

    if name in ["infant_fixation", "random_fixation", "center_fixation"]:
        return InfantVisionDatasets(
            root_dir=root_dir,
            transform=transform,
        )

    if name == "objects_train":
        return LabeledDatasets(
            root_dir=root_dir,
            transform=transform,
            split="train",
        )

    if name == "objects_test":
        return LabeledDatasets(
            root_dir=root_dir,
            transform=transform,
            split="test",
        )

    raise ValueError(f"Unhandled dataset name: {name}")
