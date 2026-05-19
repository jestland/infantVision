import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from data_aug.dataloader import build_dataset
from models.simclr import ResNetSimCLR
from simclrbuilder import SimCLR


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="./data", type=str)

    parser.add_argument(
        "--dataset-model-train",
        default="infant_fixation",
        choices=[
            "infant_fixation",
            "random_fixation",
            "center_fixation",
            "objects_train",
        ],
    )

    parser.add_argument(
        "--dataset-projection-train",
        default="objects_train",
        choices=["objects_train"],
    )

    parser.add_argument(
        "--dataset-test",
        default="objects_test",
        choices=["objects_test"],
    )

    parser.add_argument("--crop-size", default=128, type=int)
    parser.add_argument("-a", "--arch", default="resnet18")
    parser.add_argument("-j", "--workers", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("--gpu-index", default=0, type=int)

    # Match paper appendix defaults
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--temperature", default=0.08, type=float)

    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--out-dim", default=128, type=int)
    parser.add_argument("--log-every-n-steps", default=100, type=int)

    return parser.parse_args()


def set_seed(seed):
    if seed is None:
        cudnn.deterministic = False
        cudnn.benchmark = True
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_device(args):
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_index}")
        torch.cuda.set_device(device)
        return device

    return torch.device("cpu")


def make_loader(dataset, args, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=args.device.type == "cuda",
        drop_last=drop_last,
    )


def main():
    args = parse_args()
    args.data = Path(args.data)
    args.device = get_device(args)

    set_seed(args.seed)

    model_train_dataset = build_dataset(
        name=args.dataset_model_train,
        data_root=args.data,
        crop_size=args.crop_size,
    )

    projection_train_dataset = build_dataset(
        name=args.dataset_projection_train,
        data_root=args.data,
        crop_size=args.crop_size,
    )

    test_dataset = build_dataset(
        name=args.dataset_test,
        data_root=args.data,
        crop_size=args.crop_size,
    )

    model_train_loader = make_loader(
        model_train_dataset,
        args=args,
        shuffle=True,
        drop_last=True,
    )

    projection_train_loader = make_loader(
        projection_train_dataset,
        args=args,
        shuffle=True,
        drop_last=False,
    )

    test_loader = make_loader(
        test_dataset,
        args=args,
        shuffle=False,
        drop_last=False,
    )

    model = ResNetSimCLR(
        base_model=args.arch,
        out_dim=args.out_dim,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=0,
    )

    simclr = SimCLR(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
    )

    simclr.train(
        model_train_loader,
        projection_train_loader,
        test_loader,
    )


if __name__ == "__main__":
    main()
