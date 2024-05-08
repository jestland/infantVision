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

def random_crop(image):
    height, width = image.shape[:2]
    max_x = width - 128
    max_y = height - 128
    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)
    end_x_128 = start_x + 128
    end_y_128 = start_y + 128
    end_x_64 = start_x + 64
    end_y_64 = start_y + 64
    return image[start_y:end_y_64, start_x:end_x_64], image[start_y:end_y_128, start_x:end_x_128]

def main():
    infantId = '18742'
    start_index = 3711
    end_index = start_index + 19017
    csv_file_path = './data/fixation sheet/child_20170322_18742.csv'

    df = pd.read_csv(csv_file_path, header=5)
    X = 'porX'
    Y = 'porY'
    specific_data_X = extract_specific_column_range(df, X, start_index, end_index)
    specific_data_Y = extract_specific_column_range(df, Y, start_index, end_index)
    X_location = np.array(specific_data_X)
    Y_location = np.array(specific_data_Y)
    location = list(zip(X_location, Y_location))

    for i, (x, y) in enumerate(location):
        image_file_path = './data/'+ infantId +'/img_' + str(i+1) + '.jpg'
        image = read_image_file(image_file_path)

        image64, image128 = random_crop(image)
        savepath64 = os.path.join('./data/random cropping/64x64/'
                                + infantId + '/', f'{i+1}.jpg')
        savepath128 = os.path.join('./data/random cropping/128x128/'
                                + infantId + '/', f'{i+1}.jpg')
        cv2.imwrite(savepath64, image64)
        cv2.imwrite(savepath128, image128)
        print(f"randomCropping saved: {savepath64}")
        print(f"randomCropping saved: {savepath128}")



if __name__ == "__main__":
    main()
