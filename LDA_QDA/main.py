import pickle
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib import offsetbox
import pandas as pd
import scipy.io
import csv
import scipy.misc
import os
import math
from my_LDA_QDA import My_LDA_QDA




def main():
    # ---- data 1:
    LDA_or_QDA_or_Bayes_or_naiveBayes = "Bayes_mixtureModel"  #--> LDA, QDA, Bayes, naive_Bayes, Bayes_mixtureModel
    generate_toy_data_again = False
    n_classes = 3
    multi_modal = True
    if n_classes == 2:
        color_of_classes = ["blue", "red"]
    elif n_classes == 3:
        color_of_classes = ["blue", "green", "red"]
    n_samples_per_class = [200, 200, 200]
    mean0 = [-4, 4]
    cov0 = [[10, 1], [1, 5]]
    mean1 = [5, -8]
    cov1 = [[3, 0], [0, 4]]
    mean2 = [-3, -3]
    cov2 = [[6, 1.5], [1.5, 4]]     #--> or [[6, 0], [0, 4]]
    X = np.empty((2, 0))
    y = []

    # --- generate data:
    if generate_toy_data_again:
        means = [None] * n_classes
        covariances = [None] * n_classes
        for class_index in range(n_classes):
            if class_index == 0:
                mean = mean0
                cov = cov0
            elif class_index == 1:
                mean = mean1
                cov = cov1
            elif class_index == 2:
                mean = mean2
                cov = cov2
            X_class = np.random.multivariate_normal(mean, cov, n_samples_per_class[class_index]).T
            y_class = [class_index] * n_samples_per_class[class_index]
            X = np.column_stack((X, X_class))
            y.extend(y_class)
            means[class_index] = mean
            covariances[class_index] = cov
        print(X.shape)
        save_variable(variable=X, name_of_variable="X", path_to_save='./dataset/toy/')
        save_variable(variable=y, name_of_variable="y", path_to_save='./dataset/toy/')
    else:
        means = [None] * n_classes
        covariances = [None] * n_classes
        for class_index in range(n_classes):
            if class_index == 0:
                mean = mean0
                cov = cov0
            elif class_index == 1:
                mean = mean1
                cov = cov1
            elif class_index == 2:
                mean = mean2
                cov = cov2
            means[class_index] = mean
            covariances[class_index] = cov
        X = load_variable(name_of_variable="X", path='./dataset/toy/')
        y = load_variable(name_of_variable="y", path='./dataset/toy/')
        if n_classes == 2:
            mask = np.argwhere((np.array(y)==1) | (np.array(y)==0)).ravel()
            mask = mask.astype(int)
            X = X[:, mask]
            y = (np.array(y)[mask]).ravel()
    if multi_modal:
        mask = np.argwhere((np.array(y) == 1) | (np.array(y) == 0)).ravel()
        mask = mask.astype(int)
        y = np.array(y)
        y[mask] = 0
        mask = np.argwhere(np.array(y) == 2).ravel()
        mask = mask.astype(int)
        y = np.array(y)
        y[mask] = 1
        y = y.ravel()
        color_of_classes = ["blue", "red"]


    # --- plot data and estimated classification:
    my_LDA_QDA = My_LDA_QDA(X=X, y=y)
    my_LDA_QDA.plot_data(color_of_classes=color_of_classes)
    my_LDA_QDA.plot_data_and_estimationOfSpace(means, covariances, LDA_or_QDA_or_Bayes_or_naiveBayes=LDA_or_QDA_or_Bayes_or_naiveBayes,
                                               color_of_classes=color_of_classes)

    # --- classify the instances:
    my_LDA_QDA.fit_LDA_or_QDA()
    estimated_class = my_LDA_QDA.LDA_classify(X=X)
    estimated_class = my_LDA_QDA.QDA_classify(X=X)
    # print(estimated_class)







def read_csv_file(path):
    # https://stackoverflow.com/questions/46614526/how-to-import-a-csv-file-into-a-data-array
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    # convert to numpy array:
    data = np.asarray(data)
    return data

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable


if __name__ == '__main__':
    main()