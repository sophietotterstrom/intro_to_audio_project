"""
COMP.SGN.120 Intro to Audio and Speech Processing

Project for binary classification of audio samples of vehicles (cars, trams).

@author Sophie Tötterström, sophie.totterstrom@tuni.fi, 50102822
@author Aleksi Virta, aleksi.virta@tuni.fi, K425032
"""

import os
import librosa as lb
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
import random
import numpy.linalg as la
import pickle

BASE_ADDR = "./" # assume data is in current directory, change if needed
NFFT = 512
HOP_SIZE = NFFT // 2

############################################################
############# Functions related to loading data ############
def get_class_num(class_name, reverse=False):
    """ Returns either the class number related to a class string or vise versa """
    
    if reverse == False:
        if "car" in class_name.lower(): return 0
        elif "tram" in class_name.lower(): return 1
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
    test_files = []
    for filename in filenames:
        if filename[0] != ".":  # ignore hidden files
            class_label = get_class_num(filename)
            test_files.append([f'{dir_addr}{filename}', class_label])
    return test_files

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
######### Functions related to feature extraction ##########
def feature_extraction(s, sr, nmfccs, nmels):
    """
    TODO: add description
    
    @param s:
    @param sr:
    @param nmfccs:
    @param nmels:
    """
    
    win_size = NFFT
    hop_size = win_size // 2
    mfccs = librosa.feature.mfcc(y=s, sr=sr, n_mfcc=nmfccs,
                                 n_fft=NFFT, hop_length=hop_size)
    if nmels == None:
        return mfccs
    else:
        mel = librosa.feature.melspectrogram(y=s, sr=sr, n_fft=NFFT,
                                                window='hamming', n_mels=nmels)
        rms = librosa.feature.rms(y=s, frame_length=NFFT, hop_length=hop_size)
        rms = rms.reshape((np.size(rms),))
        return mfccs, mel, rms

def get_best_feature(train_data):
    """
    Function for analyzing which feature would be best for classifying samples.
    Options include: MCFF, Mel-spectrogram, RMS
    
    @param train_data: dict containing the classes and filenames of training data
    """

    avg_feats = {} # class-tuple with 0:mfcc, 1:mel, 2:rms
    for train_class in train_data:

        # NOTE uncomment to plot random signals
        #randomlist = random.sample(range(0, len(train_data[train_class])), 5)
        j = 0
        for filename in train_data[train_class]:

            audioIn, fs = prep_signal(filename)
            mfccs, mel, rms = feature_extraction(audioIn, fs, 25, 25)

            if j == 0:  # we need to initialize the variables
                sum_mfccs = mfccs
                sum_mel = mel
                sum_rms = rms
            else:       # update variables
                sum_mfccs = sum_mfccs + mfccs
                sum_mel = sum_mel + mel
                sum_rms = sum_rms + rms

            # NOTE uncomment to plot some signals and their features
            #if j in randomlist:
                #plot_signal(train_class, audioIn, fs)
                #plot_features(train_class, mfccs, mel, rms)

            j = j+1
    
        avg_mfccs = sum_mfccs/len(train_data[train_class])
        avg_mel = sum_mel/len(train_data[train_class])
        avg_rms = sum_rms/len(train_data[train_class])
        avg_feats[train_class] = ((avg_mfccs, avg_mel, avg_rms))
        
        # NOTE uncomment to plot the average features
        #plot_features(train_class, mfcc=avg_mfccs, mel=avg_mel, rms=avg_rms)
    
    # Now we have the average values of features for both classes
    classes = [key for key in train_data]

    # NOTE MSEs per se of these three features aren't really eligible for comparison.
    # Instead, if matrix is transformed into a vector, the norm of the vector
    # differences i.e., the Euclidean distance between two vectors can be computed and compared.

    mfcc_diff = (avg_feats[classes[0]])[0]-(avg_feats[classes[1]])[0]
    mel_diff = (avg_feats[classes[0]])[1] - (avg_feats[classes[1]])[1]
    rms_diff = (avg_feats[classes[0]])[2] - (avg_feats[classes[1]])[2]
    mfcc_diff = mfcc_diff.reshape(np.size(mfcc_diff))
    mel_diff = mel_diff.reshape(np.size(mel_diff))

    plot_features("car", (avg_feats[classes[0]])[0],
                  (avg_feats[classes[0]])[1], (avg_feats[classes[0]])[2])
    plot_features("tram", (avg_feats[classes[1]])[0],
                  (avg_feats[classes[1]])[1], (avg_feats[classes[1]])[2])

    mfcc_dist = la.norm(mfcc_diff)
    mel_dist = la.norm(mel_diff)
    rms_dist = la.norm(rms_diff)
    mses = {mfcc_dist: "mfcc",mel_dist: "mel", rms_dist: "rms"}

    # Just to actually see the results.
    print(f'\nFrobenius difference between class average MFCCs: {mfcc_dist:.2f}')
    print(f'\nFrobenius difference between class average Mels: {mel_dist:.2f}')
    print(f'\nFrobenius difference between class average RMSs {rms_dist:.2f}')
    return mses.get(max(mses))

def get_mfcc_data(filenames, data_label=None):
    """ Returns the MFCC and classes of each audio signal in the

    @param filenames: list of filenames of the audio signals to be processed
    @param data_label: either None, if the class label needs to be fetched or
            the actual class label if it has already been processed
    """

    if data_label == None: given_label = False
    else: given_label = True

    # Containers for labels and respective MFCCs. The idea is to make a list of tuples.
    labels = []
    mfccs = []
    for i in range(0, len(filenames)):
        if given_label == False:
            audioIn, fs = prep_signal(filenames[i][0])
            data_label = filenames[i][1]
        else:
            audioIn, fs = prep_signal(filenames[i])
        mfcc = feature_extraction(audioIn, fs, 25, nmels=None)

        # Data collection added here
        labels.append(data_label)
        mfccs.append(mfcc)
    return list(zip(mfccs, labels))

def get_mfcc_training_data(train_data):
    """ Function used to calculate the MFCCs of training data sampels. """
    
    mfccs_with_labels = []      # Container for MFCCs and respective labels. The idea is to make a list of tuples.
    for class_label in train_data:

        # Add all the mfccs of current class label into the container.
        mfccs_with_labels = mfccs_with_labels + get_mfcc_data(train_data[class_label], data_label=get_class_num(class_label))
    return mfccs_with_labels


############################################################
########## Functions related to plotting results ##########
def plot_signal(class_label, audioIn, fs):
    """ Function used to plot an audio signal """

    plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
    plt.title(f"Fig 1: Sample from class {class_label}")
    plt.plot(np.linspace(0, len(audioIn)/fs, len(audioIn)), audioIn, 'b')
    plt.grid()
    plt.xlabel('time [s]')
    plt.show()

def plot_features(label, mfcc, mel, rms):
    """ Function used to print the features (MFCC, Mel, RMS """

    plt.figure()
    librosa.display.specshow(mfcc, x_axis='time', hop_length=HOP_SIZE)
    plt.colorbar()
    plt.title(f'Average MFCC for class {label}')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title(f"Average Mel Spectrogram for class {label}")
    librosa.display.specshow(mel, x_axis='time', hop_length=HOP_SIZE)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title(f"Average RMS for class {label}")
    plt.plot(np.arange(np.size(rms)), rms, 'b')
    plt.show()


############################################################
############ Functions related to the Classifier ############
def class_acc(pred, gt):
    """
    TODO: add description

    @param pred:
    @param gt: 
    """

    N = len(pred)
    corr_class = N

    for i in range(0, len(pred)):
        if pred[i] != gt[i]:
            corr_class = corr_class - 1

    print(f'\nClassication accuracy: {corr_class * 100 / N:.2f}%')

def classifier_1nn(sample, reference):
    """
    Simple model for 1nn classification.
    TODO: add description
    
    @param sample:
    @param reference:
    """
    optimal = - 1               # Initializing optimal class value
    smallest_distance = -1     # Current min distance to train data

    for r in reference:
        norm = la.norm(sample[0]-r[0])
        if smallest_distance == -1:
            smallest_distance = norm
            optimal = 0
        elif norm <= smallest_distance: # Compare distances
            smallest_distance = norm
            optimal = r[1] # Class value
    return optimal

def perform_classification(mfccs_train, mfccs_to_classify):
    """
    TODO: add description

    @param mfccs_train: 
    @param mfccs_to_classify: 
    """
    preds = []
    correct_classes = []
    for m in mfccs_to_classify:
        correct_classes.append(m[1])
        nn = classifier_1nn(m,mfccs_train)
        preds.append(nn)
    class_acc(preds, correct_classes)

def load_presaved(mfccs_train=None):
    """
    Function can be used to reset the loaded training data,
    or to simply return the presaved data if mffcs_train=None. 
    """

    if mfccs_train != None:
        output = open('weigths.pkl', 'wb')
        pickle.dump(mfccs_train, output)
        output.close()
    
    pkl_file = open('weigths.pkl', 'rb')
    mfccs_train_presaved = pickle.load(pkl_file)
    pkl_file.close()
    return mfccs_train_presaved

def main():

    # NOTE uncomment to load data
    #train_data, test_data_filenames, validation_data_filenames = load_data_filenames()

    # NOTE uncomment to perform analysis on which feature is the best for classifying. Our result is MFCC (see report for further details).
    # best_feat = get_best_feature(train_data)
    
    # NOTE uncomment to load the MFCCs of test, training, and validation data.
    #mfccs_train = get_mfcc_training_data(train_data)
    #mfccs_test = get_mfcc_data(test_data_filenames)
    #mfccs_validation = get_mfcc_data(validation_data_filenames)

    # NOTE uncomment to perform classification
    # Change around the mfccs_to_classify variable to test model with different data
    #perform_classification(mfccs_train=mfccs_train_presaved, mfccs_to_classify=mfccs_test)
    
    ############################################################
    ######### This is what was asked to be submitted! ##########

    mfccs_train_presaved = load_presaved()                                  # load presaved values ("weigths") 
    filename = 'car8.wav'                                                   # the example value to be classified
    test_files = []                                                         # made functions work for a list of data
    test_files.append([f'{BASE_ADDR}{filename}', get_class_num(filename)])
    mfcc_test = get_mfcc_data(test_files)                                   # get MFCC for classification
    perform_classification(mfccs_train=mfccs_train_presaved, mfccs_to_classify=mfcc_test)
    
    ############################################################
    

if __name__ == "__main__":
    main()