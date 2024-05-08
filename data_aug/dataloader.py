from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms

transform_to_64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


transform_to_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class LabeledDatasets(Dataset):
    def __init__(self, root_dir, transform=transform_to_64):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # self.labels = pd.read_csv(root_dir/exp_12_dictionary.csv)
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


class InfantVisionDatasets(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_paths = []
        self.fname_paths = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            self.fname_paths.append(folder_path)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(folder_path, img_name))

    def __len__(self):
        return len(self.image_paths) - 1

    def __getitem__(self, idx):
        img1_path = self.image_paths[idx]
        img2_path = self.image_paths[idx + 1]
        fname1_path = os.path.dirname(os.path.dirname(img1_path))
        fname2_path = os.path.dirname(os.path.dirname(img2_path))

        try:
            assert fname1_path == fname2_path
        except AssertionError:
            print(f"AssertionError: {fname1_path} != {fname2_path}")

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)


        return img1, img2, fname1_path.replace(self.root_dir, '')

# shift #
root_dir = 'data/shift/'
# # Origin #
# root_dir = 'data/origin/'
# # Remove #
# root_dir = 'data/remove/'


dataset_plainBackground = LabeledDatasets(root_dir=root_dir.join('plain background/'), transform=transform_to_64)
dataset_objectsFixation = LabeledDatasets(root_dir=root_dir.join('objects fixation/'), transform=transform_to_64)
dataset_infantFixation64 = InfantVisionDatasets(root_dir=root_dir.join('fixation cropping/64x64/'), transform=None)
dataset_infantFixation128 = InfantVisionDatasets(root_dir=root_dir.join('fixation cropping/128x128/'), transform=None)
dataset_randomFixation64 = InfantVisionDatasets(root_dir=root_dir.join('random cropping/64x64/'), transform=None)
dataset_randomFixation128 = InfantVisionDatasets(root_dir=root_dir.join('random cropping/128x128/'), transform=None)
dataset_centerFixation64 = InfantVisionDatasets(root_dir=root_dir.join('center cropping/64x64/'), transform=None)
dataset_centerFixation128 = InfantVisionDatasets(root_dir=root_dir.join('center cropping/128x128/'), transform=None)
dataset_centerFixation240 = InfantVisionDatasets(root_dir=root_dir.join('center cropping/240x240/'), transform=None)
dataset_centerFixation480 = InfantVisionDatasets(root_dir=root_dir.join('center cropping/480x480/'), transform=None)


dataloader_plainBackground = DataLoader(dataset_plainBackground, batch_size=256., num_workers=8, shuffle=False)
dataloader_objectsFixation = DataLoader(dataset_objectsFixation, batch_size=256, num_workers=8, shuffle=False)
dataloader_infantFixation64 = DataLoader(dataset_infantFixation64, batch_size=256, num_workers=8, shuffle=True)
dataloader_infantFixation128 = DataLoader(dataset_infantFixation128, batch_size=256, num_workers=8, shuffle=True)
dataloader_randomFixation64 = DataLoader(dataset_randomFixation64, batch_size=256, num_workers=8, shuffle=True)
dataloader_randomFixation128 = DataLoader(dataset_randomFixation128, batch_size=256, num_workers=8, shuffle=True)
dataloader_centerFixation64 = DataLoader(dataset_centerFixation64, batch_size=256, num_workers=8, shuffle=True)
dataloader_centerFixation128 = DataLoader(dataset_centerFixation128, batch_size=256, num_workers=8, shuffle=True)
dataloader_centerFixation240 = DataLoader(dataset_centerFixation240, batch_size=256, num_workers=8, shuffle=True)
dataloader_centerFixation480 = DataLoader(dataset_centerFixation480, batch_size=256, num_workers=8, shuffle=True)



# def main():
#     for batch_idx, (imgs1, imgs2) in enumerate(dataloader_infantFixation64):
#         print(f"Batch {batch_idx}:")
#         print(f"  imgs1 shape: {imgs1.shape}, dtype: {imgs1.dtype}")
#         print(f"  imgs2 shape: {imgs2.shape}, dtype: {imgs2.dtype}")
#
#         if batch_idx > 2:
#             break
#
#
# if __name__ == "__main__":
#     main()
