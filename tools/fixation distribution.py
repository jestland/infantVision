import pandas as pd
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns

def read_csv_file(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

def extract_specific_column_range(df, column_name, start_index, end_index):
    specific_data = df.loc[start_index:end_index, column_name]
    return specific_data

def read_image_file(image_file_path):
    image = cv2.imread(image_file_path)
    return image



def main():
    infantId = '16963'
    start_index = 2704
    end_index = start_index + 18929
    csv_file_path = 'data/fixation sheet/child_20160209_17358.csv'

    df = pd.read_csv(csv_file_path, header=5)
    X = 'porX'
    Y = 'porY'
    specific_data_X = extract_specific_column_range(df, X, start_index, end_index)
    specific_data_Y = extract_specific_column_range(df, Y, start_index, end_index)
    X_location = np.array(specific_data_X)
    Y_location = np.array(specific_data_Y)
    mask = (X_location >= 0) & (Y_location >= 0)
    X_location = X_location[mask]
    Y_location = Y_location[mask]
    # contour_levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1]

    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=X_location, y=Y_location, fill=True, cmap="viridis", cbar=True)
    plt.title("Fixation distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.xlim(0, 1000)  #
    # plt.ylim(0, 1000)  #
    plt.show()





if __name__ == "__main__":
    main()
