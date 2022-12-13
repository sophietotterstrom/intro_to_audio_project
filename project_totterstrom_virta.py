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
from sklearn.metrics import mean_squared_error

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

    audio_files = []
    class_files = {}

    dirs = os.listdir(dirs_addr)    # lists different directories (car, tram)
    for dir in dirs:
        if dir[0] != ".":           # ignore hidden directories
            
            dir_files = os.listdir(f'{dirs_addr}{dir}/')
            #class_num = get_class_num(dir)
            for file in dir_files:
                if file[0] != ".":
                    #audio_files[f'{dirs_addr}{dir}/{file}'] = class_num
                    audio_files.append(f'{dirs_addr}{dir}/{file}')
            
            class_files[dir] = audio_files
            audio_files = []
    
    return class_files

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
    validation_files = get_files(f'{BASE_ADDR}validationdata/')

    return train_files, test_files, validation_files


def prep_signal(filename):
    """ Loads the audio signal and resizes it (and padds with zeros if needed) """

    audioIn, fs=lb.load(filename, sr=22050)
    audioIn = audioIn[:int(5*fs)]
    audioIn = lb.util.fix_length(audioIn, size=int(5*fs))
    return audioIn, fs



############################################################
############ Functions related to the Classifier ############
def class_acc(pred, gt):
    N = len(pred)
    corr_class = N

    for i in range(0, len(pred)):
        if pred[i] != gt[i]:
            corr_class = corr_class - 1

    print(f'\nClassication accuracy: {corr_class * 100 / N:.2f}%')


## Simple model for 1nn classification.
def classifier_1nn(sample,reference):
    optimal = - 1               # Initializing optimal class value
    smallest_distance = -1      # Current min distance to train data

    for r in reference:
        norm = np.norm(sample-r)
        if norm < smallest_distance: # Compare distances
            smallest_distance = norm
            optimal = reference[r]
    return optimal


############################################################
######### Functions related to feature extraction ##########
def feature_extraction(s,sr,nfft,nmfccs,nmels):
    win_size = nfft
    hop_size = win_size // 2
    mfccs = librosa.feature.mfcc(y=s, sr=sr, n_mfcc=nmfccs,
                                 n_fft=nfft, hop_length=hop_size)
    mel = librosa.feature.melspectrogram(y=s, sr=sr, n_fft=nfft,
                                         window='hamming', n_mels=nmels)
    rms = librosa.feature.rms(y=s, frame_length=nfft,hop_length=hop_size)
    rms = rms.reshape((np.size(rms),))
    return mfccs, mel, rms

def get_best_feature(train_data, hop_size, nfft):

    avg_feats = {} # class-tuple with 0:mfcc, 1:mel, 2:rms

    for train_class in train_data:

        j = 0
        randomlist = random.sample(range(0, len(train_data[train_class])), 5)
        for filename in train_data[train_class]:

            audioIn, fs = prep_signal(filename)
            mfccs, mel, rms = feature_extraction(audioIn, fs, nfft, 25, 25)

            if j == 0:
                sum_mfccs = mfccs
                sum_mel = mel
                sum_rms = rms
            else:
                sum_mfccs = sum_mfccs + mfccs
                sum_mel = sum_mel + mel
                sum_rms = sum_rms + rms

            #if j in randomlist:
                #plot_signal(train_class, audioIn, fs)
                #plot_features(hop_size, train_class, mfccs, mel, rms)

            j = j+1
    
        avg_mfccs = sum_mfccs/len(train_data[train_class])
        avg_mel = sum_mel/len(train_data[train_class])
        avg_rms = sum_rms/len(train_data[train_class])

        avg_feats[train_class] = ((avg_mfccs, avg_mel, avg_rms))

        #plot_features(hop_size, train_class, mfcc=avg_mfccs, mel=avg_mel, rms=avg_rms)
    
    # Now we have the average values of features for both classes
    classes = [key for key in train_data]
                                    # first class avg mffc                      # second class avg mffc
    mffc_mse = mean_squared_error((avg_feats[classes[0]])[0], (avg_feats[classes[1]])[0])
    mel_mse = mean_squared_error((avg_feats[classes[0]])[1], (avg_feats[classes[1]])[1])
    rms_mse = mean_squared_error((avg_feats[classes[0]])[2], (avg_feats[classes[1]])[2])
    # TODO return the feature associated with the biggest mse


############################################################
########## Functions related to plotting results ##########
def plot_signal(class_label, audioIn, fs):
    plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
    plt.title(f"Fig 1: Sample from class {class_label}")
    plt.plot(np.linspace(0, len(audioIn)/fs, len(audioIn)), audioIn, 'b')
    plt.grid()
    plt.xlabel('time [s]')
    plt.show()

def plot_features(hop_size, label, mfcc, mel, rms):
    plt.figure()
    librosa.display.specshow(mfcc, x_axis='time', hop_length=hop_size)
    plt.colorbar()
    plt.title(f'Average MFCC for class {label}')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title(f"Average Mel Spectrogram for class {label}")
    librosa.display.specshow(np.log10(mel), x_axis='time', hop_length=hop_size)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title(f"Average RMS for class {label}")
    plt.plot(np.arange(np.size(rms)), rms, 'b')
    plt.show()
    #print(np.size(np.arange(np.size(rms))))
    #print(np.size(rms))
    #print("break")



def main():

    nfft = 512
    hop_size = nfft // 2
    train_data, test_data_filenames, validation_data_filenames = load_data_filenames()

    get_best_feature(train_data, hop_size=hop_size, nfft=nfft)


if __name__ == "__main__":
    main()