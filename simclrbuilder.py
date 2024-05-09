import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, save_checkpoint
from utils import generate_embeddings, accuracy
from torch.linalg import lstsq
from loss import infantVision_Loss

torch.manual_seed(42)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    # def info_nce_loss(self, features):
    #
    #     labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(self.args.device)
    #
    #     features = F.normalize(features, dim=1)
    #
    #     similarity_matrix = torch.matmul(features, features.T)
    #
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    #
    #     logits = torch.cat([positives, negatives], dim=1)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
    #
    #     logits = logits / self.args.temperature
    #     return logits, labels

    def train(self, model_train_loader, projection_train_loader, test_loader):

        scaler = GradScaler()
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            self.model.train()
            for img1, img2 in tqdm(model_train_loader):
                images = torch.cat((img1, img2), dim=0)
                images.to(self.args.device)
                with autocast():
                    # features = self.model(images)
                    # logits, labels = self.info_nce_loss(features)
                    # loss = self.criterion(logits, labels)
                    representation, projection = self.model(images)
                    projection, pair = projection.split(self.args.batch_size)
                    loss = infantVision_Loss(self.args.batch_size, self.args.temperature)(projection, pair)

                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            self.model.eval()
            with torch.no_grad():
                if n_iter % self.args.log_every_n_steps == 0:
                    train_embeddings, train_labels = generate_embeddings(self.model, projection_train_loader)
                    test_embeddings, test_labels = generate_embeddings(self.model, test_loader)
                    lstsq_model = lstsq(train_embeddings, F.one_hot(train_labels, 24).type(torch.float32))
                    acc = ((test_embeddings @ lstsq_model.solution).argmax(dim=-1) == test_labels).sum() / len(
                        test_embeddings)
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', acc, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            # if epoch_counter >= 10:
            #     self.scheduler.step()
        logging.info("Training has finished.")

        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
