"""
COMP.SGN.120 Intro to Audio and Speech Processing

@author Sophie Tötterström
@author Aleksi Virta
"""

import os
import librosa as lb
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy import signal

BASE_ADDR = "./" # assume data is in current directory, change if needed

############################################################
############# Functions related to loading data ############
def get_class_num(class_name, reverse=False):
    """ Returns either the class number related to a class string or vise versa """
    
    if reverse == False:
        if class_name == "car": return 0
        elif class_name == "tram": return 1
    else:
        if class_name == 0: return "car"
        elif class_name == 1: return "tram"

def get_dir_files(dirs_addr):
    """ Returns filenames and classes of files in multiple 
    directories under the specified directory"""

    audio_files = {}
    dirs = os.listdir(dirs_addr)    # lists different directories (car, tram)'¨
    for dir in dirs:
        if dir[0] != ".":           # ignore hidden directories
            print(dir)
            dir_files = os.listdir(f'{dirs_addr}{dir}/')
            class_num = get_class_num(dir)
            for file in dir_files:
                if file[0] != ".":
                    audio_files[f'{dirs_addr}{dir}/{file}'] = class_num
    return audio_files

def get_files(dir_addr):
    """ Returns filenames and classes of files in given directory """

    filenames = os.listdir(dir_addr)
    test_files = {}
    for filename in filenames:
        test_files[filename] = get_class_num(filename.lower())

def load_data_filenames():
    """ Loads the audio data: filenames to a map with their associated class labels """

    train_files = get_dir_files(f'{BASE_ADDR}traindata/')
    test_files = get_files(f'{BASE_ADDR}testdata/')
    validation_files = get_dir_files(f'{BASE_ADDR}validationdata/')

    return train_files, test_files, validation_files



############################################################
############ Functions related to the Classifier ############
def class_acc(pred, gt):
    N = len(pred)
    corr_class = N

    for i in range(0, len(pred)):
        if pred[i] != gt[i]:
            corr_class = corr_class - 1

    print(f'\nClassication accuracy: {corr_class * 100 / N:.2f}%')

def feature_extraction(s,sr,nfft,nmfccs,nmels):
    win_size = nfft
    hop_size = win_size // 2
    mfccs = librosa.feature.mfcc(y=s, sr=sr, n_mfcc=nmfccs,
                                 n_fft=nfft, hop_length=hop_size)
    mel = librosa.feature.melspectrogram(y=s, sr=sr, n_fft=nfft,
                                         window='hamming', n_mels=nmels)
    return mfccs, mel

def classifier_1nn(signal, training_data, training_labels):
    # Initializing the index for the optimal image and
    # (square of) the minimum distance.
    opt_ind = -1
    min_dist = 0

    for i in range(training_data.shape[0]):

        # If the index is zero, update the distance and the optimal index.
        # Otherwise update only if the square distance is smaller than the
        # previous square.
        if i == 0 or np.sum((x - trdata[i]) ** 2) < min_dist:
            min_dist = np.sum((x - trdata[i]) ** 2)
            opt_ind = i
    return training_labels[opt_ind]


def main():
    nfft = 512
    hop_size = nfft // 2
    train_data_filenames, test_data_filenames, validation_data_filenames = load_data_filenames()


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

            mfccs, mel = feature_extraction(audioIn, fs, nfft, 30, 30)
            plt.figure()
            librosa.display.specshow(mfccs, x_axis='time', hop_length=hop_size)
            plt.show()
            plt.figure()
            librosa.display.specshow(np.log10(mel), x_axis='time', hop_length=hop_size)
            plt.show()
            print(sample_class)

        j = j+1


if __name__ == "__main__":
    main()