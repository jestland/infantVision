import os
import random
from shutil import copy2


def data_set_split(src_data_folder, target_data_folder, train_scale=0.8):
    print("start to split data set...")
    class_names = os.listdir(src_data_folder)
    # create train, test folder
    split_names = ['train', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        current_idx = 0
        train_num = 0
        test_num = 0
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                # print("{}copy to{}".format(src_img_path, train_folder))
                train_num = train_num + 1
            else:
                copy2(src_img_path, test_folder)
                # print("{}copy to{}".format(src_img_path, test_folder))
                test_num = test_num + 1

            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print("Training set{}：{}".format(train_folder, train_num))
        print("Test set{}：{}".format(test_folder, test_num))


if __name__ == '__main__':
    src_data_folder = "./data/objects fixation/"
    target_data_folder = "./data/shift/objects fixation/"
    data_set_split(src_data_folder, target_data_folder)