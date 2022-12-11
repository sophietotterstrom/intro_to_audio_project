"""
COMP.SGN.120 Intro to Audio and Speech Processing

@author Sophie Tötterström
@author Aleksi Virta
"""

import os
import librosa as lb
from matplotlib import pyplot as plt
import numpy as np
import random

BASE_ADDR = "./" # let's assume data is in current directory

def get_class_num(class_name, reverse=False):
    """ Returns either the class number related to a class string or vise versa """
    
    if reverse == False:
        if class_name == "bus": return 0
        elif class_name == "car": return 1
        elif class_name == "tram": return 2
    else:
        if class_name == 0 : return "bus"
        elif class_name == 1: return "car"
        elif class_name == 2: return "tram"

def load_data_filenames():
    """ Loads train and test data filenames to a map with their associated class labels """

    audio_train_files = {}
    train_dirs = os.listdir(f'{BASE_ADDR}traindata/')
    for dir in train_dirs:
        if dir[0] != ".":   # ignore hidden directories
            dir_audio_train_files = os.listdir(f'{BASE_ADDR}traindata/{dir}/')
            class_num = get_class_num(dir)
            for file in dir_audio_train_files:
                if file[0] != ".":
                    audio_train_files[f'{BASE_ADDR}traindata/{dir}/{file}'] = class_num

    audio_test_filenames = os.listdir(f'{BASE_ADDR}testdata/')
    audio_test_files = {}
    for filename in audio_test_filenames:
        if "bus" in filename.lower(): audio_test_files[filename] = 0
        elif "car" in filename.lower(): audio_test_files[filename] = 1
        elif "tram" in filename.lower(): audio_test_files[filename] = 2

    return audio_train_files, audio_test_files

def main():
    
    train_data_filenames, test_data_filenames = load_data_filenames()


    j = 0
    randomlist = random.sample(range(0, len(train_data_filenames)), 10)
    for filename in train_data_filenames:
        if j in randomlist:
            sample_class = get_class_num(train_data_filenames[filename], reverse=True)

            audioIn, fs=lb.load(filename, sr=None)


            plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
            plt.title(f"Fig 1: Sample from class {sample_class}")
            plt.plot(np.linspace(0,len(audioIn) / fs,len(audioIn)),audioIn, 'b')
            plt.grid()
            plt.xlabel('time [s]')
            plt.show()

        j = j+1


if __name__ == "__main__":
    main()