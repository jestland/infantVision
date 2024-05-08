import argparse
import torch
import torch.backends.cudnn as cudnn
from data_aug.dataloader import dataset_infantFixation64, train_dataset_objectsFixation64, test_dataset_objectsFixation64
from models.simclr import ResNetSimCLR
from simclrbuilder import SimCLR
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('-data', default='./data')
# parser.add_argument('--data_organizing', default='shift', type=str, choices=['origin', 'shift', 'remove']
parser.add_argument('-dataset_model_train', default='dataset_infantFixation64',
                    choices=['dataset_plainBackground64', 'dataset_plainBackground128',
                             'dataset_objectsFixation64', 'dataset_objectsFixation128',
                             'dataset_randomFixation64', 'dataset_randomFixation128',
                             'dataset_centerFixation64', 'dataset_centerFixation128',
                             'dataset_centerFixation240', 'dataset_centerFixation480',
                             'dataset_infantFixation64', 'dataset_infantFixation128'])
parser.add_argument('-dataset_projection_train', default='dataset_objectsFixation64',
                    choices=['dataset_objectsFixation64', 'dataset_objectsFixation128'])
parser.add_argument('-dataset_test', default='dataset_objectsFixation64',
                    choices=['dataset_objectsFixation64', 'dataset_plainBackground128'])
parser.add_argument('-a', '--arch', default='resnet18')
parser.add_argument('-j', '--workers', default=8)
parser.add_argument('--epochs', default=100)
parser.add_argument('-b', '--batch-size', default=256)
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr', choices=[0.0001, 0.0005, 0.001, 0.01, 0.1, 0.5])
parser.add_argument('--wd', '--weight-decay', default=0.000005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay', choices=[0.000001, 0.000005, 0.00001, 0.0001, 0.001, 0.005])
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)', choices=[0.05, 0.08, 0.1, 0.2, 0.5, 1])
# parser.add_argument('--skip_frame', default=0, type=float,
#                     help='skip select number of frames', choices=[0, 5, 10, 20, 30, 60, 90, 120])


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

    if args.dataset_model_train == 'dataset_infantFixation64':
        model_train_dataset = dataset_infantFixation64
    if args.dataset_projection_train == 'dataset_objectsFixation64':
        projection_train_dataset = train_dataset_objectsFixation64
    if args.dataset_test == 'dataset_objectsFixation64':
        test_dataset = test_dataset_objectsFixation64


    model_train_loader = DataLoader(model_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    projection_train_loader = DataLoader(projection_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # criterion = torch.nn.CrossEntropyLoss().to(args.device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(model_train_dataset), eta_min=0,
                                                           last_epoch=-1)

    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(model_train_loader, projection_train_loader, test_loader)



if __name__ == "__main__":
    main()
