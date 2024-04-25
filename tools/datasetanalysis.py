import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def read_csv_file(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

def extract_specific_column_range(df, column_name, start_index, end_index):
    specific_data = df.loc[start_index:end_index, column_name]
    return specific_data




def main():
    csv_file_path = 'data\infantVisionDataInfo.csv'
    plainbackground_path = 'data\plain background'
    objectsfixation_path = 'data\objects fixation'
    fixation64_path = 'data/fixation cropping/64x64'
    fixation128_path = 'data/fixation cropping/128x128'


    df = pd.read_csv(csv_file_path)
    infantID = np.array(df['infantID'])
    startFrameNumber = np.array(df['startFrameNumber'])
    frameDataSize = np.array(df['frameDataSize'])
    croppingDataSize_64 = np.array(df['croppingDataSize_64'])
    croppingDataSize_128 = np.array(df['croppingDataSize_128'])
    randomCroppingDataSize_64 = np.array(df['randomCroppingDataSize_64'])
    randomCroppingDataSize_128 = np.array(df['randomCroppingDataSize_128'])
    startTime = np.array(df['startTime'])
    endTime = np.array(df['endTime'])
    startPorX = np.array(df['startPorX'])
    startPorY = np.array(df['startPorY'])
    infantAge = np.array(df['infantAge'])
    gender = np.array(df['gender'])

    plainbackgroundSize = len(os.listdir(plainbackground_path))

    objectsfixation = os.listdir(objectsfixation_path)
    objectsfixationSize = []
    for content in objectsfixation:
        if os.path.isdir(objectsfixation_path+'/'+content):
            objectsfixationSize.append(len(os.listdir(objectsfixation_path+'/'+content)))


    # size of datasets
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.get_cmap('Blues')
    x = np.array(['Plain background', 'Objects fixation', 'Random fixation',
                  'Center fixation below 480x480','Center fixation 480x480',
                  'infant fixation 64x64', 'infant fixation 128x128', 'raw frames'])
    y = np.array([plainbackgroundSize, sum(objectsfixationSize), sum(randomCroppingDataSize_64),
                  sum(randomCroppingDataSize_64), sum(randomCroppingDataSize_64)-12466-9151-10801-8566,
                  sum(croppingDataSize_64), sum(croppingDataSize_128), sum(frameDataSize)])
    bar = plt.bar(x, y, 0.5, color=colors(0.4), edgecolor='grey')
    plt.bar_label(bar, label_type='edge')
    ax.set_title('Size of datasets', fontsize=12, color='black', alpha=0.7)
    ax.set_ylabel("", fontsize=12, color='black', alpha=0.7, rotation=360)
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y', useMathText=True)
    plt.xticks(rotation=15)
    plt.savefig('project/infantVision/results/data analysis/size of datasets.pdf', pad_inches=0)
    plt.show()

    # cropping Propotion
    # fig, ax = plt.subplots(figsize=(22, 6))
    # colors = plt.get_cmap('Blues')
    # colors_64 = plt.get_cmap('Reds')
    # colors_128 = plt.get_cmap('Purples')
    # x = [str(x) for x in infantID]
    # y_frameDataSize = frameDataSize/frameDataSize
    # y_croppingDataSize_64 = croppingDataSize_64/frameDataSize
    # y_croppingDataSize_128 = croppingDataSize_128/frameDataSize
    # bar = plt.bar(x, y_frameDataSize, 0.5, color=colors(0.4), edgecolor='grey')
    # bar = plt.bar(x, y_croppingDataSize_64, 0.5, color=colors_64(0.4), edgecolor='grey')
    # bar = plt.bar(x, y_croppingDataSize_128, 0.5, color=colors_128(0.4), edgecolor='grey')
    # ax.set_title('Infant fixation proportion after cropping', fontsize=12, color='black', alpha=0.7)
    # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y', useMathText=True)
    # plt.legend(['Raw frame', 'Cropping size: 64x64', 'Cropping size: 128x128'], frameon=False,bbox_to_anchor=(1,1))
    # plt.yticks(np.arange(0, 1.2, 0.2), [f'{i}%' for i in range(0, 120, 20)])
    # plt.xlabel('infant ID')
    # plt.xticks(rotation=30)
    # plt.savefig('project/infantVision/results/data analysis/Infant fixation proportion after cropping.pdf', pad_inches=0)
    # plt.show()

    # plain dataset
    # fig, ax = plt.subplots(figsize=(13, 6))
    # colors = plt.get_cmap('Greens')
    # x = ['helmet','house','bluecar','rose','elephant','snowman','rabbit','spongebob',
    #      'turtle','hammer','ladybug','mantis','greencar','saw','doll','phone',
    #      'rubiks','shovel','bigwheels','whitecar','ladybugstick','purpleblock','bed','clearblock']
    # y = [plainbackgroundSize/len(x)] * len(x)
    # bar = plt.bar(x, y, 0.7, color=colors(0.4), edgecolor='grey')
    # ax.set_title('Plain dataset', fontsize=12, color='black', alpha=0.7)
    # plt.bar_label(bar, label_type='edge')
    # plt.xlabel('labels')
    # plt.xticks(rotation=30)
    # plt.savefig('project/infantVision/results/data analysis/Plain dataset.pdf', pad_inches=0)
    # plt.show()

    # # Objects fixation dataset
    # fig, ax = plt.subplots(figsize=(13, 6))
    # colors = plt.get_cmap('Greens')
    # x = ['helmet','house','bluecar','rose','elephant','snowman','rabbit','spongebob',
    #      'turtle','hammer','ladybug','mantis','greencar','saw','doll','phone',
    #      'rubiks','shovel','bigwheels','whitecar','ladybugstick','purpleblock','bed','clearblock']
    # y = objectsfixationSize
    # bar = plt.bar(x, y, 0.7, color=colors(0.4), edgecolor='grey')
    # ax.set_title('Objects fixation dataset', fontsize=12, color='black', alpha=0.7)
    # plt.bar_label(bar, label_type='edge')
    # plt.xlabel('labels')
    # plt.xticks(rotation=30)
    # ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y', useMathText=True)
    # plt.savefig('project/infantVision/results/data analysis/Object fixation dataset.pdf', pad_inches=0)
    # plt.show()



if __name__ == "__main__":
    main()
