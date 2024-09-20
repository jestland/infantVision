import torch
from torchvision import transforms

def get_transformations(img, crop_size=128, flag=True):
    if flag:
        s = 1.0
        # normalize = transforms.Normalize(mean=rgb_mean, std=rgb_std)
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        gaussian_blur = transforms.GaussianBlur((3, 3), (0.1, 2.0))
        gaussian_noise = transforms.GaussianNoise(std=0.1)
        train_transform = transforms.Compose([
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.08, 0.2)), # 随机裁剪和缩放
            # transforms.RandomApply([color_jitter], p=0.8),
            # transforms.RandomApply([gaussian_blur], p=0.8),
            # transforms.RandomApply([gaussian_noise], p=0.8),
            transforms.ToTensor(),
        ])
        #
        # val_transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        img_transformed = train_transform(img)
        return img_transformed
    else:
        return img
