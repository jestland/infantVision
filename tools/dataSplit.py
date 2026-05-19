import os
import random
from shutil import copy2

def data_set_split(src_data_folder, target_data_folder, train_scale=0.8):
    print("start to split data set...")

    class_names = [d for d in os.listdir(src_data_folder)
                   if os.path.isdir(os.path.join(src_data_folder, d))]

    for split_name in ['train', 'test']:
        for class_name in class_names:
            os.makedirs(os.path.join(target_data_folder, split_name, class_name), exist_ok=True)

    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = [f for f in os.listdir(current_class_data_path)
                            if f.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(current_all_data)

        train_folder = os.path.join(target_data_folder, 'train', class_name)
        test_folder = os.path.join(target_data_folder, 'test', class_name)

        split_idx = int(len(current_all_data) * train_scale)
        train_files = current_all_data[:split_idx]
        test_files = current_all_data[split_idx:]

        for f in train_files:
            copy2(os.path.join(current_class_data_path, f), train_folder)
        for f in test_files:
            copy2(os.path.join(current_class_data_path, f), test_folder)

        print(f"{'='*20} {class_name} {'='*20}")
        print(f"Train: {train_folder} — {len(train_files)} images")
        print(f"Test:  {test_folder} — {len(test_files)} images")

if __name__ == '__main__':
    src_data_folder = "./data/objects fixation/"
    target_data_folder = "./data/shift/objects fixation/"
    data_set_split(src_data_folder, target_data_folder)
