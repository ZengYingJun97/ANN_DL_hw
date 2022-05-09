from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_barest_cancer_datasets():
    print("Start load barest_cancer datasets...")
    barest_cancer = load_breast_cancer()

    data = pd.DataFrame(barest_cancer.data, columns=barest_cancer.feature_names)
    data['class'] = barest_cancer.target

    X = data[data.columns.drop('class')]
    Y = data['class']
    print("X: {}, Y: {}".format(np.size(X, 0), np.size(Y, 0)))

    transfer = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    print("X_train: {}, Y_train: {}, X_test: {}, Y_test: {}".format(np.size(X_train, 0), np.size(Y_train, 0), np.size(X_test, 0), np.size(Y_test, 0)))
    print("End load barest_cancer datasets...")
    return transfer.fit_transform(X_train), transfer.fit_transform(X_test), Y_train, Y_test, len(Counter(barest_cancer.target))

def get_visualization_barest_cancer(X=None, Y=None):
    bc = load_breast_cancer()
    pca = PCA(2)
    if X is None:
        projected = pca.fit_transform(bc.data)
    else:
        projected = pca.fit_transform(X)

    if Y is None:
        c = bc.target
    else:
        c = Y
    plt.scatter(projected[:, 0], projected[:, 1], c=c, edgecolor='none',
                alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

def get_features_name():
    bc = load_breast_cancer()
    return bc.feature_names