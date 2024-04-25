from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from numpy.random import randint
import torch
import random
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),  #(64.64), (128,128)
    transforms.ToTensor(),
])



class LabeledDatasets(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for idx, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path)

        # if self.transform:
        #     image = self.transform(image)

        return image, label

class InfantVisionDatasets(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(folder_path, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image





# dataset_plainBackground = LabeledDatasets(root_dir='data/plain background/', transform=transform)
# dataset_objectsFixation = LabeledDatasets(root_dir='data/objects fixation/', transform=transform)

dataset_infantFixation64 = InfantVisionDatasets(root_dir='data/fixation cropping/64x64/', transform=transform)
# dataset_infantFixation128 = InfantVisionDatasets(root_dir='data/fixation cropping/128x128', transform=transform)

# dataset_randomFixation64 = InfantVisionDatasets(root_dir='data/random cropping/64x64', transform=transform)
# dataset_randomFixation128 = InfantVisionDatasets(root_dir='data/fandom cropping/128x128', transform=transform)

# dataset_centerFixation64 = InfantVisionDatasets(root_dir='data/center cropping/64x64', transform=transform)
# dataset_centerFixation128 = InfantVisionDatasets(root_dir='data/center cropping/128x128', transform=transform)
# dataset_centerFixation240 = InfantVisionDatasets(root_dir='data/center cropping/240x240', transform=transform)
# dataset_centerFixation480 = InfantVisionDatasets(root_dir='data/center cropping/480x480', transform=transform)

# dataloader_plainBackground = DataLoader(dataset_plainBackground, batch_size=128, num_workers=0, shuffle=False)
# dataloader_objectsFixation = DataLoader(dataset_objectsFixation, batch_size=128, num_workers=0, shuffle=False)

dataloader_infantFixation64 = DataLoader(dataset_infantFixation64, batch_size=128, num_workers=0, shuffle=False)
# dataloader_infantFixation128 = DataLoader(dataset_infantFixation128, batch_size=128, num_workers=0, shuffle=False)
#
# dataloader_randomFixation64 = DataLoader(dataset_randomFixation64, batch_size=128, num_workers=0, shuffle=False)
# dataloader_randomFixation128 = DataLoader(dataset_randomFixation128, batch_size=128, num_workers=0, shuffle=False)
#
# dataloader_centerFixation64 = DataLoader(dataset_centerFixation64, batch_size=128, num_workers=0, shuffle=False)
# dataloader_centerFixation128 = DataLoader(dataset_centerFixation128, batch_size=128, num_workers=0, shuffle=False)
# dataloader_centerFixation240 = DataLoader(dataset_centerFixation240, batch_size=128, num_workers=0, shuffle=False)
# dataloader_centerFixation480 = DataLoader(dataset_centerFixation480, batch_size=128, num_workers=0, shuffle=False)


# for batch in dataloader_infantFixation64:
#     images, labels = batch
#     print("Images shape:", images.shape)  # (batch_size, channels, height, width)
#     print("Labels:", labels)
#     break


positive_pairs = []
negative_pairs = []

data_list = [data for data in dataloader_infantFixation64]

for i in range(len(data_list) - 1):
    positive_pairs.append((data_list[i], data_list[i + 1]))

for i in range(len(data_list)):
    for j in range(len(data_list)):
        if abs(i - j) > 1:
            negative_pairs.append((data_list[i], data_list[j]))

print("Number of positive pairs:", len(positive_pairs))
print("Number of negative pairs:", len(negative_pairs))


