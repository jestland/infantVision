import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.dataloader import InfantVisionDatasets, dataset_infantFixation64
from models.simclr import ResNetSimCLR
from simclrbuilder import SimCLR
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-data', default='./data')
parser.add_argument('-dataset-name', default='dataset_infantFixation64',
                    choices=['dataset_plainBackground', 'dataset_objectsFixation',
                             'dataset_randomFixation64', 'dataset_randomFixation128',
                             'dataset_centerFixation64', 'dataset_centerFixation128',
                             'dataset_centerFixation240', 'dataset_centerFixation480',
                             'dataset_infantFixation64', 'dataset_infantFixation128'])
parser.add_argument('-a', '--arch', default='resnet18')
parser.add_argument('-j', '--workers', default=12)
parser.add_argument('--epochs', default=100)
parser.add_argument('-b', '--batch-size', default=128)

parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)', choices=[0.07, 0.08, 0.09, 0.1])


def main():
    args = parser.parse_args()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    train_dataset = dataset_infantFixation64

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
