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

# Global variables
BASE_ADDR = "./" # assume data is in current directory, change if needed
NFFT = 512
HOP_SIZE = NFFT // 2

############################################################
############# Functions related to loading data ############
def get_class_num(class_str, reverse=False):
    """ 
    Returns either the class number related to a class string or vise versa 
    """
    
    if reverse == False:
        if "car" in class_str.lower(): return 0
        elif "tram" in class_str.lower(): return 1
    else:
        if class_str == 0: return "car"
        elif class_str == 1: return "tram"

def get_dir_files(dirs_addr):
    """ 
    Function returns filenames and classes of files in directories 
    @param dir_addr: the parent directory to be checked for sub directories and files 
    """

    audio_files = []
    class_files = {}

    dirs = os.listdir(dirs_addr)    # lists different directories (car, tram)
    for dir in dirs:
        if dir[0] != ".":           # ignore hidden directories
            
            dir_files = os.listdir(f'{dirs_addr}{dir}/')
            for file in dir_files:
                if file[0] != ".":  # ignore hidden files
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


############################################################
######### Functions related to feature extraction ##########
def prep_signal(filename):
    """ 
    Loads the audio signal and resizes it (and padds with zeros if needed).

    @param filename: the name of the file from which signal is to be read
    """

    audioIn, fs=lb.load(filename, sr=22050)
    audioIn = audioIn[:int(5*fs)]                           # limit signal to 5 seconds
    audioIn = lb.util.fix_length(audioIn, size=int(5*fs))   # if shorter, zero-padd to be 5 seconds
    return audioIn, fs

def feature_extraction(s, sr, nmfccs, nmels):
    """
    A function for extracting desired features of a given sample.
    
    @param s: A sample for which the data will be extracted.
    @param sr: Integer value sample rate used.
    @param nmfccs: Integer value number of MFCCs.
    @param nmels: Integer value number of Mel filters.
    """
    
    win_size = NFFT
    hop_size = win_size // 2

    # The MFCC
    mfccs = librosa.feature.mfcc(y=s, sr=sr, n_mfcc=nmfccs,
                                 n_fft=NFFT, hop_length=hop_size)

    # If nmels isn't specified, just return the mfccs.
    if nmels == None:
        return mfccs

    # Otherwise compute the Mels and RMS and return the values.
    else:
        mel = librosa.feature.melspectrogram(y=s, sr=sr, n_fft=NFFT,
                                                window='hamming', n_mels=nmels)
        rms = librosa.feature.rms(y=s, frame_length=NFFT, hop_length=hop_size)
        rms = rms.reshape((np.size(rms),))
        return mfccs, mel, rms

def get_best_feature(train_data):
    """
    Function for analyzing which feature would be best for classifying samples.
    Options include: MFCC, Mel-spectrogram, RMS
    
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
    """ 
    Returns the MFCC and classes of each audio signal from given filenames.

    @param filenames: list of filenames of the audio signals to be processed
    @param data_label: either None, if the class label needs to be fetched or
            the actual class label if it has already been processed
    """

    # Find if we are using predetermined label
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
    
    # Container for MFCCs and respective labels. The idea is to make a list of tuples.
    mfccs_with_labels = []      
    for class_label in train_data:

        # Add all the mfccs of current class label into the container.
        mfccs_with_labels = mfccs_with_labels + get_mfcc_data(train_data[class_label], data_label=get_class_num(class_label))
    return mfccs_with_labels


############################################################
########## Functions related to plotting results ##########
def plot_signal(class_label, audioIn, fs):
    """ 
    Function used to plot an audio signal 
    
    @param class_label: the class the sample belongs in
    @param audioIn: the audio signal to be plotted
    @param fs: the sampling rate"""

    plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
    plt.title(f"Fig 1: Sample from class {class_label}")
    plt.plot(np.linspace(0, len(audioIn)/fs, len(audioIn)), audioIn, 'b')
    plt.grid()
    plt.xlabel('time [s]')
    plt.show()

def plot_features(label, mfcc, mel, rms):
    """ 
    Function used to print the features (MFCC, Mel, RMS) of an audio signal
    """

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
    A function for computing the classification accuracy based on
    given lists of predicted values and the actual values.

    @param pred: list of predicted class labels.
    @param gt: list of correct class labels.
    """

    N = len(pred)
    corr_class = N

    for i in range(0, len(pred)):

        # If the class labels don't match, decrease of the number of correct
        # predictions.
        if pred[i] != gt[i]:
            corr_class = corr_class - 1

    print(f'\nClassification accuracy: {corr_class * 100 / N:.2f}%')

def classifier_1nn(sample, reference):
    """
    A nearest neighbour classifier that classifies a sample looking at
    the nearest item in the reference data.
    
    @param sample: An array of values. In our case most likely a matrix.
    @param reference: A list of tuples where the tuples have the reference
    element as the first value and class label as the second.
    """
    optimal = - 1               # Initializing optimal class value
    smallest_distance = -1     # Current min distance to train data

    for r in reference:
        norm = la.norm(sample[0]-r[0])

        # When computing the first distance, automatically update the
        # smallest distance to this one. Note that this also updates
        # updates the predicted value.
        if smallest_distance == -1:
            smallest_distance = norm
            optimal = r[1]
        elif norm <= smallest_distance: # Compare distances
            smallest_distance = norm
            optimal = r[1] # Class value
    return optimal

def perform_classification(mfccs_train, mfccs_to_classify, positive=1):
    """
    A function that does the classification of signals based on their MFCCs
    by looking at the nearest neighbour of each MFCC in the training data.

    @param mfccs_train: A list of tuples where the tuple has MFCC as a first
    element and the corresponding class as the second. This is the list of
    training data.
    @param mfccs_to_classify: A list of tuples where the tuple has MFCC as a first
    element and the corresponding class as the second. This is the list that is
    to be classified.
    @param positive: An integer defining which class will be regarder as the
    "positive" in the classification in order to compute recall and precision.
    """

    # Initializing containers and counter values for positive samples.
    preds = []
    correct_classes = []
    predicted_positives = 0
    true_positives = 0
    all_observations = 0
    for m in mfccs_to_classify:
        correct_classes.append(m[1])
        if m[1] == positive:
            all_observations = all_observations + 1 # Update the total observations.
        prediction = classifier_1nn(m,mfccs_train)

        # If the prediction matches the positive class label, update the number
        # of positive predictions.
        if prediction == positive:
            predicted_positives = predicted_positives + 1

        # If the prediction is a true positive, update the corresponding number.
        if prediction == positive and m[1] == positive:
            true_positives = true_positives + 1
        preds.append(prediction)

    # Get classification metrics.
    class_acc(preds, correct_classes)
    print(f'\nClassification precision: {true_positives/predicted_positives*100:.2f}%')
    print(f'\nClassification recall: {true_positives/all_observations*100:.2f}%')

def load_presaved(mfccs_train=None):
    """
    Function can be used to return the presaved data, or reset the loaded data.

    @param mfccs_train: if other than None, load values again for the training data
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
    #mfccs_train_presaved = load_presaved()
    #perform_classification(mfccs_train=mfccs_train_presaved, mfccs_to_classify=mfccs_validation)

    ############################################################
    ######### This is what was asked to be submitted! ##########

    mfccs_train_presaved = load_presaved()                                  # load presaved values ("weigths")
    filename = 'car8.wav'                                                   # the example value to be classified
    test_files = []                                                         # made functions work for a list of data
    test_files.append([f'{BASE_ADDR}{filename}', get_class_num(filename)])
    mfcc_test = get_mfcc_data(test_files)                                   # get MFCC for classification

    # positive=0 (car) since it was chosen to use car sample (else effects precision/recall unreasonably)
    perform_classification(mfccs_train=mfccs_train_presaved, mfccs_to_classify=mfcc_test, positive=0)
    
    ############################################################
    

if __name__ == "__main__":
    main()