import torch
import torchvision.transforms as T

class Augment:

   def __init__(self, img_size, s=1):
       color_jitter = T.ColorJitter(
           0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
       )
       # 10% of the image
       gaussian_blur = T.GaussianBlur((3, 3), (0.1, 2.0))

       self.train_transform = torch.nn.Sequential(
           T.Normalize(mean=[0.431, 0.488, 0.403], std=[0.186, 0.124, 0.167]),
           T.RandomGrayscale(p=0.2),
           T.RandomResizedCrop(size=img_size),
           # T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
           # T.RandomApply([color_jitter], p=0.8),
           # T.RandomApply([gaussian_blur], p=0.5),
       )

   def __call__(self, x):
       return self.train_transform(x)