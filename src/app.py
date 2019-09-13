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

#Altered version of remove noise
def RemoveNoise(df, activityVal):
    # df = pd.read_csv('dataset/dataset_1.txt', sep=',', header=None)
    # Butterworth low-pass filter. You could try different parameters and other filters.
    b, a = signal.butter(4, 0.04, 'high', analog=False)
    df_clean = df[df[24] == activityVal].values
    for i in range(3):
        df_clean[:,i] = signal.lfilter(b, a, df_clean[:, i])
    return df_clean

if __name__ == '__main__':
    data = "../dataset/"
    dataset=LoadData(os.getcwd()+"/../dataset/raw/")
    #VisualiseData(dataset)
    print("Hello")
    print(dataset)
