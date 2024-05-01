from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from numpy.random import randint
import torch
import random
from torchvision import datasets, transforms
from main import parser
from itertools import combinations

transform_to_64 = transforms.Compose([
    transforms.Resize((64, 64)),  #(64.64), (128,128)
    transforms.ToTensor(),
])
#
# transform_to_128 = transforms.Compose([
#     transforms.Resize((128, 128)),  #(64.64), (128,128)
#     transforms.ToTensor(),
# ])

args = parser.parse_args()

def prepare_pairs(dataset):
    dataset_length = len(dataset)
    positive_pairs = []
    negative_pairs = []

    for i in range(dataset_length - 1):
        positive_pairs.append((dataset[i], dataset[i + 1]))

    all_combinations = list(combinations(range(dataset_length), 2))
    negative_pairs = [(dataset[i], dataset[j]) for i, j in all_combinations if abs(i - j) > 1]

    return positive_pairs, negative_pairs


class LabeledDatasets(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.n_classses = 24
        # self.n_length = 1536

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


class InfantVisionDatasetsBuilder(Dataset):
    def __init__(self, images, root_dir, transform=None):
        self.images = images
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
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

class InfantVisionDatasets(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]



# dataset_plainBackground = LabeledDatasets(root_dir='data/plain background/', transform=transform_128)
dataset_objectsFixation = LabeledDatasets(root_dir='data/objects fixation/', transform=transform_to_64)
dataset_infantFixation64 = InfantVisionDatasetsBuilder(root_dir='data/fixation cropping/64x64/', transform=None)
# dataset_infantFixation128 = InfantVisionDatasetsBuilder(root_dir='data/fixation cropping/128x128', transform=None)
# dataset_randomFixation64 = InfantVisionDatasetsBuilder(root_dir='data/random cropping/64x64', transform=None)
# dataset_randomFixation128 = InfantVisionDatasetsBuilder(root_dir='data/fandom cropping/128x128', transform=None)
# dataset_centerFixation64 = InfantVisionDatasetsBuilder(root_dir='data/center cropping/64x64', transform=None)
# dataset_centerFixation128 = InfantVisionDatasetsBuilder(root_dir='data/center cropping/128x128', transform=None)
# dataset_centerFixation240 = InfantVisionDatasetsBuilder(root_dir='data/center cropping/240x240', transform=None)
# dataset_centerFixation480 = InfantVisionDatasetsBuilder(root_dir='data/center cropping/480x480', transform=None)
positive_pairs, negative_pairs = prepare_pairs(dataset_infantFixation64)
positive_labels = [1] * len(positive_pairs)
negative_labels = [0] * len(negative_pairs)
all_pairs = positive_pairs + negative_pairs
all_labels = positive_labels + negative_labels
dataset_infantFixation64 = InfantVisionDatasets(all_pairs, all_labels)

# dataloader_plainBackground = DataLoader(dataset_plainBackground, batch_size=256., num_workers=8, shuffle=False)
dataloader_objectsFixation = DataLoader(dataset_objectsFixation, batch_size=256, num_workers=8, shuffle=False)
dataloader_infantFixation64 = DataLoader(dataset_infantFixation64, batch_size=256, num_workers=8, shuffle=True)
# dataloader_infantFixation128 = DataLoader(dataset_infantFixation128, batch_size=256, num_workers=8, shuffle=True)
#
# dataloader_randomFixation64 = DataLoader(dataset_randomFixation64, batch_size=256, num_workers=8, shuffle=True)
# dataloader_randomFixation128 = DataLoader(dataset_randomFixation128, batch_size=256, num_workers=8, shuffle=True)
#
# dataloader_centerFixation64 = DataLoader(dataset_centerFixation64, batch_size=256, num_workers=8, shuffle=True)
# dataloader_centerFixation128 = DataLoader(dataset_centerFixation128, batch_size=256, num_workers=8, shuffle=True)
# dataloader_centerFixation240 = DataLoader(dataset_centerFixation240, batch_size=256, num_workers=8, shuffle=True)
# dataloader_centerFixation480 = DataLoader(dataset_centerFixation480, batch_size=256, num_workers=8, shuffle=True)



# pair_dataloader = DataLoader(pair_dataset, batch_size=256, shuffle=True)
# for (img_pair, label) in pair_dataloader:
#     print("Image Pair:", img_pair, "Label:", "Positive" if label == 1 else "Negative")





