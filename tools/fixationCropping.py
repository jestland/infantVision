import pandas as pd
import numpy as np
import random
import cv2
import os

def read_csv_file(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

def extract_specific_column_range(df, column_name, start_index, end_index):
    specific_data = df.loc[start_index:end_index, column_name]
    return specific_data

def read_image_file(image_file_path):
    image = cv2.imread(image_file_path)
    return image

def fixation_crop(image, center_x, center_y, crop_size):
    start_x = int(center_x - crop_size[0] / 2)
    start_y = int(center_y - crop_size[1] / 2)
    end_x = int(center_x + crop_size[0] / 2)
    end_y = int(center_y + crop_size[1] / 2)
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

def main():
    infantId = '18742'
    start_index = 3711
    end_index = start_index + 19017
    csv_file_path = 'E:/project/infantVision/data/fixation sheet/child_20170322_18742.csv'
    croppingSize = 128

    df = pd.read_csv(csv_file_path, header=5)
    X = 'porX'
    Y = 'porY'
    crop_size = (croppingSize, croppingSize)
    specific_data_X = extract_specific_column_range(df, X, start_index, end_index)
    specific_data_Y = extract_specific_column_range(df, Y, start_index, end_index)
    X_location = np.array(specific_data_X)
    Y_location = np.array(specific_data_Y)
    location = list(zip(X_location, Y_location))
    imgIndex = 0
    for i, (x, y) in enumerate(location):
        image_file_path = 'E:/project/infantVision/data/'+ infantId +'/img_' + str(i+1) + '.jpg'
        image = read_image_file(image_file_path)

        fixation_cropped_image = fixation_crop(image, x, y, crop_size)
        if fixation_cropped_image.shape[0] != croppingSize or fixation_cropped_image.shape[1] != croppingSize:
            continue
        imgIndex+=1
        savepath_fixation = os.path.join('E:/project/infantVision/data/fixation cropping/'
                                + str(croppingSize) + 'x' + str(croppingSize) +'/'
                                + infantId + '/', f'{imgIndex}.jpg')
        cv2.imwrite(savepath_fixation, fixation_cropped_image)
        print(f"fixationCropping saved: {savepath_fixation}")


if __name__ == "__main__":
    main()
