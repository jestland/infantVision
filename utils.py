import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


def generate_embeddings(model, dataloader, device):
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for images, y in dataloader:
            images = images.to(device, non_blocking=True)

            embedding, _ = model(images)
            embedding = F.normalize(embedding.flatten(start_dim=1), dim=1)

            embeddings.append(embedding.cpu())
            labels.append(y.cpu())

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

    if is_best:
        best_path = os.path.join(os.path.dirname(filename), "model_best.pth.tar")
        shutil.copyfile(filename, best_path)


def save_config_file(model_checkpoints_folder, args):
    model_checkpoints_folder = Path(model_checkpoints_folder)
    model_checkpoints_folder.mkdir(parents=True, exist_ok=True)

    config_path = model_checkpoints_folder / "config.yml"

    args_dict = vars(args).copy()
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
        elif isinstance(value, torch.device):
            args_dict[key] = str(value)

    with open(config_path, "w") as outfile:
        yaml.safe_dump(args_dict, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(correct_k.mul(100.0 / batch_size))

        return results
