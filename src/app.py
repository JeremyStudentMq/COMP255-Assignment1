import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os

#a simple function for loading data and return dataframes for each set
#i want to be able to seperate the data into different folders e.g. "raw" untouched
#data, "clean" modified data etc.
#The point is I want to be able to point to a folder and have the function load
#the datasets within there as there own functions

def LoadData(dataFolderpath):
    dfs = []
    for file in os.listdir(dataFolderpath):
        if ".txt" in file:
            df = pd.read_csv(dataFolderpath+file, sep=',', header=None)
            dfs.append(df)
    return dfs

def CleanNoise(dfs):
    cleanDfs=[]
    for df in dfs:
        for c in range(1, 14):
            activity_data = df[df[24] == c].values
            b, a = signal.butter(4, 0.04, 'low', analog=False)
            for j in range(24):
                activity_data[:, j] = signal.lfilter(b, a, activity_data[:, j])
        cleanDfs.append(activity_data)
    return cleanDfs
def VisualiseData(dfs):
    for df in dfs:
        for i in range(1,13):
            df_plot = df[df[24] == i].values
            # In this example code, only accelerometer 1 data (column 1 to 3) is used
            plt.plot(df_plot[:, 0:9])
            plt.show()

'''
For raw sensor data, it usually contains noise that arises from different sources, such as sensor mis-
calibration, sensor errors, errors in sensor placement, or noisy environments. We could apply filter to remove noise of sensor data
to smooth data. In this example code, Butterworth low-pass filter is applied.
'''


#Pre - The data must be noise filtered already
def GetFeatures(dfs):
    # mean, median, mode Arithmetic average, median and mode
    # var, std Variance and standard deviation
    # sem Standard error of mean
    # skew Sample skewness
    variance = SumOfTermsMinusMeanPow(terms,sampleMeans,2)/sizeOfTerms-1
    stdDev = sqrt((1/size)*SumOfTermsMinusMeanPow(terms,sampleMeans,2)))
    standardError = stdDev/sqrt(n)
    skewness = CalculateSkewness(terms,mean,stdDev,size)
    kurtosis=  CalculateKurtosis(terms,mean,stdDev,size)

def CalculateKurtosis(terms,mean,stdDev,size):
    tmpKurto=0
    for term in terms:
        tmpKurto=tmpKurto+((term-mean)^4/size*stdDev^4)
    return tmpKurto

def CalculateSkewness(terms,mean,stdDev,size):
    tmpSkew=0
    for term in terms:
        tmpSkew=tmpSkew+((term-mean)^3/size*stdDev^3)
    return tmpSkew

def SumOfTermsMinusMeanPow(terms,sampleMean,pow):
    tmpSum = 0
    for term in terms:
        tmpSum = tmpSum+(term-sampleMean)^pow
    return tmpSum

def CreateTrainingAndTestingDataSets(dfs):
        for df in dfs:

            datat_len = len(df)
            activity_data=df.values
            training_len = math.floor(datat_len * 0.8)
            training_data = activity_data[:training_len, :]
            testing_data = activity_data[training_len:, :]

            # data segementation: for time series data, we need to segment the whole time series, and then extract features from each period of time
            # to represent the raw data. In this example code, we define each period of time contains 1000 data points. Each period of time contains
            # different data points. You may consider overlap segmentation, which means consecutive two segmentation share a part of data points, to
            # get more feature samples.
            training_sample_number = training_len // 1000 + 1
            testing_sample_number = (datat_len - training_len) // 1000 + 1


#Altered sample version of remove noise for my purposes
def RemoveNoise(df, activityVal):
    b, a = signal.butter(4, 0.04, 'high', analog=False)
    df_clean = df[df[24] == activityVal].values
    for i in range(3):
        df_clean[:,i] = signal.lfilter(b, a, df_clean[:, i])
    return df_clean

#1. load dataset -> done
#2. visualize data -> done
#3. remove signal noises -> done
#4. extract features -> done
#5. prepare training set -> to do
#6. training the given models -> to do
#7. test the given models -> to do
#8. print out the evaluation results -> to do


if __name__ == '__main__':
    data = "../dataset/"
    dataset=LoadData(os.getcwd()+"/../dataset/raw/")
    #VisualiseData(dataset)
