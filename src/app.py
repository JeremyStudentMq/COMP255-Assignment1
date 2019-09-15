import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, precision_score, recall_score,precision_recall_fscore_support
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
        cleanFrame = np.empty(shape=(0,24))
        for c in range(1, 14):
            activity_data = df[df[24]==c].values
            b, a = signal.butter(4, 0.04, 'low', analog=False)
            for j in range(24):
                activity_data[:, j] = signal.lfilter(b, a, activity_data[:, j])
            cleanFrame= cleanFrame.append
    cleanDfs.append(activity_data)
    return cleanDfs
def VisualiseData(dfs):
    for df in dfs:
        for i in range(1,14):
            df_plot = df[df[24] == i].values
            # In this example code, only accelerometer 1 data (column 1 to 3) is used
            plt.plot(df_plot[:, 0:9])
            plt.show()

def loadDfFromPathWithArg(path,arg):
    frames=[]
    for file in os.listdir(path):
        if arg in file:
            df = pd.read_csv(path+file, sep=',', header=None)
            df= df.iloc[1:]
            df = df.drop(df.columns[0],axis=1)
            frames.append(df)
    dfs=pd.concat(frames)
    #dfs = dfs.iloc[1:]
    #dfs = dfs.drop(dfs.columns[0],axis=1)
    return dfs

def model_training_and_testing():
    df_training=loadDfFromPathWithArg(os.getcwd()+"/../dataset/dirty/",'training')
    df_testing=loadDfFromPathWithArg(os.getcwd()+"/../dataset/dirty/",'testing')
    print(df_training)
    y_train = df_training[193].values
    # Labels should start from 0 in sklearn
    y_train = y_train - 1
    df_training = df_training.drop([193], axis=1)
    X_train = df_training.values
    y_test = df_testing[193].values
    y_test = y_test - 1
    df_testing = df_testing.drop([193], axis=1)
    X_test = df_testing.values
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall: ', recall_score(y_test, y_pred,average='weighted'))
    print('F Score: ', precision_recall_fscore_support(y_test, y_pred, average='weighted'))
    # We could use confusion matrix to view the classification for each activity.
    print(confusion_matrix(y_test, y_pred))

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2, 1e-3, 1e-4],
                     'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 100]},
                    {'kernel': ['linear'], 'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}]
    acc_scorer = make_scorer(accuracy_score)
    grid_obj  = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring=acc_scorer)
    grid_obj  = grid_obj .fit(X_train, y_train)
    clf = grid_obj.best_estimator_
    print('best clf:', clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall: ', recall_score(y_test, y_pred,average='weighted'))
    print('F Score: ', precision_recall_fscore_support(y_test, y_pred, average='weighted'))
    print(confusion_matrix(y_test, y_pred))

'''
For raw sensor data, it usually contains noise that arises from different sources, such as sensor mis-
calibration, sensor errors, errors in sensor placement, or noisy environments. We could apply filter to remove noise of sensor data
to smooth data. In this example code, Butterworth low-pass filter is applied.
'''

def GetFeatures(input_data,data_size):
    output_set=np.empty(shape=(0,193))
    sample_data=[]
    input_Data = input_data[:data_size,:]
    for s in range(data_size):
        if s < data_size - 1:
            sample_data = input_data[1000*s:1000*(s + 1), :]
        else:
            sample_data = input_data[1000*s:, :]

        #I'm getting all.
        feature_sample = []
        for i in range(24): #all monitors used
            activity_data=sample_data[:, i]
            activity_data = RemoveNoise(activity_data,'low', 0.04)
            min=np.min(activity_data)
            max=np.max(activity_data)
            mean=np.mean(activity_data)
            terms=activity_data
            size = len(activity_data)
            variance = SumOfTermsMinusMeanPow(terms,mean,2)/size-1
            stdDev = math.sqrt((1/size)*SumOfTermsMinusMeanPow(terms,mean,2))
            standardError = stdDev/math.sqrt(size)
            skewness = CalculateSkewness(terms,mean,stdDev,size)
            kurtosis=  CalculateKurtosis(terms,mean,stdDev,size)
            feature_sample.append(min)
            feature_sample.append(max)
            feature_sample.append(mean)
            feature_sample.append(variance)
            feature_sample.append(stdDev)
            #print(stdDev)
            feature_sample.append(standardError)
            feature_sample.append(skewness)
            feature_sample.append(kurtosis)
            #print(feature_sample)
        feature_sample.append(sample_data[0, -1])
        feature_sample = np.array([feature_sample])
        output_set = np.concatenate((output_set, feature_sample), axis=0)
    return output_set

    # mean, median, mode Arithmetic average, median and mode
    # var, std Variance and standard deviation
    # sem Standard error of mean
    # skew Sample skewness


def CalculateKurtosis(terms,mean,stdDev,size):
    tmpKurto=0
    for term in terms:
        tmpKurto=tmpKurto+(((term-mean)**4))
    return tmpKurto/(size-1*stdDev**4)

def CalculateSkewness(terms,mean,stdDev,size):
    tmpSkew=0
    for term in terms:
        tmpSkew=tmpSkew+(((term-mean)**3))
    return tmpSkew/(size-1*stdDev**3)

def SumOfTermsMinusMeanPow(terms,sampleMean,pow):
    tmpSum = 0
    for term in terms:
        tmpSum = tmpSum+(term-sampleMean)**pow
    return tmpSum

def CreateTrainingAndTestingDataSets(dfs):
        output_sets = []
        i=1
        for df in dfs: #deal with each dataset
            training_features = np.empty(shape=(0,193))
            testing_features = np.empty(shape=(0,193))
            for c in range(1,14): #deal with each activity
                activity_data=df[df[24]==c].values
                datat_len = len(activity_data)
                training_len = math.floor(datat_len * 0.8)
                training_data = activity_data[:training_len, :]
                testing_data = activity_data[training_len:, :]
                training_sample_number = training_len // 1000 + 1
                training_features=np.concatenate((training_features,GetFeatures(training_data,training_sample_number)),axis=0)
                testing_sample_number = (datat_len - training_len) // 1000 + 1
                testing_features=np.concatenate((testing_features,GetFeatures(testing_data,testing_sample_number)),axis=0)
                #print("Feature Appended")
            #print("Saved : " +"training_data"+str(i)+".csv")
            #print("Saved : " +"testing_data"+str(i)+".csv")
            save_training_df=pd.DataFrame(training_features)
            save_testing_df=pd.DataFrame(testing_features)
            save_training_df.to_csv(os.getcwd()+"/../dataset/dirty/"+"training_data"+str(i)+".csv")
            save_testing_df.to_csv(os.getcwd()+"/../dataset/dirty/"+"testing_data"+str(i)+".csv")
            i=i+1
                #print(training_features)
                #print(testing_features)
#Altered sample version of remove noise for my purposes
def RemoveNoise(data,filterType, alpha):
    b, a = signal.butter(4,alpha, filterType, analog=False)
    data = signal.lfilter(b, a, data)
    return data

#
# def RemoveNoise(data):
#     b, a = signal.butter(4, 0.04, 'high', analog=False)
#     data = signal.lfilter(b, a, data)
#     return data

#1. load dataset -> done
#2. visualize data -> done
#3. remove signal noises -> done
#4. extract features -> done
#5. prepare training set -> done
#6. training the given models -> done
#7. test the given models -> done
#8. print out the evaluation results -> done

# HOW TO RUN
'''
There needs to be a dataset directory above the folder that app.py is run from.
It needs to be called dataset and it MUST contain the following directories
    - "raw" - which contains the raw .txt files
    - "dirty" - which is where the feature data is output.

so there needs to be
Root
    - src
        -app.py
    -dataset
        -raw
            -dataset.txt
        -dirty
            - feature samples will be stored here
'''
#
if __name__ == '__main__':
    data = "../dataset/"
    datasets=LoadData(os.getcwd()+"/../dataset/raw/")
    #VisualiseData(datasets)
    CreateTrainingAndTestingDataSets(datasets)
    model_training_and_testing()
    #VisualiseData(dataset)
