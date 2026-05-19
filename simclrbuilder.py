import logging
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.linalg import lstsq
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss import infantVision_Loss
from tools.augmentations import get_transformations
from utils import save_config_file, save_checkpoint, generate_embeddings


class SimCLR:
    def __init__(self, *, model, optimizer, scheduler, args):
        self.args = args
        self.model = model.to(args.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = SummaryWriter()
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, "training.log"),
            level=logging.INFO,
        )

    def train(self, model_train_loader, projection_train_loader, test_loader):
        use_cuda = self.args.device.type == "cuda"
        scaler = GradScaler(enabled=use_cuda)

        save_config_file(self.writer.log_dir, self.args)

        global_step = 0
        logging.info(f"Start training for {self.args.epochs} epochs.")
        logging.info(f"Device: {self.args.device}")

        for epoch in range(self.args.epochs):
            self.model.train()

            progress = tqdm(
                model_train_loader,
                desc=f"Epoch [{epoch + 1}/{self.args.epochs}]",
            )

            for batch in progress:
                img1, img2 = batch[0], batch[1]

                img1 = get_transformations(img1, flag=True)
                img2 = get_transformations(img2, flag=True)

                img1 = img1.to(self.args.device, non_blocking=True)
                img2 = img2.to(self.args.device, non_blocking=True)

                batch_size = img1.size(0)
                images = torch.cat((img1, img2), dim=0)

                with autocast(enabled=use_cuda):
                    _, projection = self.model(images)
                    z_i, z_j = projection[:batch_size], projection[batch_size:]

                    criterion = infantVision_Loss(
                        batch_size=batch_size,
                        temperature=self.args.temperature,
                    ).to(self.args.device)

                    loss = criterion(z_i, z_j)

                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                global_step += 1

                if global_step % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar(
                        "train/loss",
                        loss.item(),
                        global_step=global_step,
                    )
                    self.writer.add_scalar(
                        "train/lr",
                        self.scheduler.get_last_lr()[0],
                        global_step=global_step,
                    )

                progress.set_postfix(loss=f"{loss.item():.4f}")

            self.scheduler.step()

            acc = self.evaluate_linear_probe(
                projection_train_loader,
                test_loader,
            )

            self.writer.add_scalar(
                "eval/linear_probe_acc",
                acc,
                global_step=epoch + 1,
            )

            logging.info(
                f"Epoch {epoch + 1}/{self.args.epochs}, "
                f"loss={loss.item():.4f}, acc={acc:.4f}"
            )

        self.save_checkpoint()

    def evaluate_linear_probe(self, projection_train_loader, test_loader):
        self.model.eval()

        with torch.no_grad():
            train_embeddings, train_labels = generate_embeddings(
                self.model,
                projection_train_loader,
                device=self.args.device,
            )
            test_embeddings, test_labels = generate_embeddings(
                self.model,
                test_loader,
                device=self.args.device,
            )

            train_embeddings = train_embeddings.to(self.args.device)
            test_embeddings = test_embeddings.to(self.args.device)
            train_labels = train_labels.to(self.args.device)
            test_labels = test_labels.to(self.args.device)

            num_classes = int(train_labels.max().item()) + 1
            y_train = F.one_hot(
                train_labels,
                num_classes=num_classes,
            ).float()

            solution = lstsq(train_embeddings, y_train).solution
            predictions = (test_embeddings @ solution).argmax(dim=-1)

            acc = (predictions == test_labels).float().mean().item()

        return acc

    def save_checkpoint(self):
        checkpoint_name = f"checkpoint_{self.args.epochs:04d}.pth.tar"
        checkpoint_path = os.path.join(self.writer.log_dir, checkpoint_name)

        save_checkpoint(
            {
                "epoch": self.args.epochs,
                "arch": self.args.arch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            is_best=False,
            filename=checkpoint_path,
        )

        logging.info(
            f"Model checkpoint and metadata saved at {self.writer.log_dir}."
        )
