import pandas as pd
import numpy as np
import random
import cv2
import os


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
    infantID = ['16963','17275','17358','17402','17406','17527','17565',
                '17592','17608','17662','17718','17757','17782','17843',
                '17848','17874','17878','17919','17933','18068','18100',
                '18419','18431','18459','18625','18742','18796','18996',
                '19357','19505','19536','19544','19615','19694','19812',
                '19859','19954','20510','21015']


    for id in infantID:
        infantId = id
        croppingSize64 = 64
        croppingSize128 = 128
        croppingSize480 = 240

        id_path = 'E:/project/infantVision/data/' + infantId
        crop_size64 = (croppingSize64, croppingSize64)
        crop_size128 = (croppingSize128, croppingSize128)
        crop_size480 = (croppingSize480, croppingSize480)

        dataSize = len(os.listdir(id_path))
        imgIndex = 0
        for i in range(dataSize):
            image_file_path = 'E:/project/infantVision/data/'+ infantId +'/img_' + str(i+1) + '.jpg'
            image = read_image_file(image_file_path)
            x = image.shape[1] / 2
            y = image.shape[0] / 2
            fixation_cropped_image64 = fixation_crop(image, x, y, crop_size64)
            fixation_cropped_image128 = fixation_crop(image, x, y, crop_size128)
            fixation_cropped_image480 = fixation_crop(image, x, y, crop_size480)

            imgIndex+=1
            savepath_fixation64 = os.path.join('E:/project/infantVision/data/center cropping/'
                                    + str(croppingSize64) + 'x' + str(croppingSize64) +'/'
                                    + infantId + '/', f'{imgIndex}.jpg')
            savepath_fixation128 = os.path.join('E:/project/infantVision/data/center cropping/'
                                             + str(croppingSize128) + 'x' + str(croppingSize128) + '/'
                                             + infantId + '/', f'{imgIndex}.jpg')
            savepath_fixation480 = os.path.join('E:/project/infantVision/data/center cropping/'
                                             + str(croppingSize480) + 'x' + str(croppingSize480) + '/'
                                             + infantId + '/', f'{imgIndex}.jpg')
            cv2.imwrite(savepath_fixation64, fixation_cropped_image64)
            cv2.imwrite(savepath_fixation128, fixation_cropped_image128)
            cv2.imwrite(savepath_fixation480, fixation_cropped_image480)

            print(f"fixationCropping saved: {savepath_fixation64},{savepath_fixation128},{savepath_fixation480}")

            print(f"fixationCropping saved:{savepath_fixation480}")

if __name__ == "__main__":
    main()
