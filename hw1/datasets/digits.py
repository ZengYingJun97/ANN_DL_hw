from collections import Counter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_digits_datasets():
    print("Start load digits datasets...")
    digits = load_digits()

    data = pd.DataFrame(digits.data, columns=digits.feature_names)
    data['class'] = digits.target

    X = data[data.columns.drop('class')]
    Y = data['class']
    print("X: {}, Y: {}".format(np.size(X, 0), np.size(Y, 0)))

    transfer = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    print("X_train: {}, Y_train: {}, X_test: {}, Y_test: {}".format(np.size(X_train, 0), np.size(Y_train, 0), np.size(X_test, 0), np.size(Y_test, 0)))
    print("End load digits datasets...")
    return transfer.fit_transform(X_train), transfer.fit_transform(X_test), Y_train, Y_test, len(Counter(digits.target))

def get_visualization_digits(X=None, Y=None):
    digits = load_digits()
    pca = PCA(2)
    if X is None:
        projected = pca.fit_transform(digits.data)
    else:
        projected = pca.fit_transform(X)

    if Y is None:
        c = digits.target
    else:
        c = Y
    plt.scatter(projected[:, 0], projected[:, 1], c=c, edgecolor='none',
                alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

def get_features_name():
    digits = load_digits()
    return digits.feature_names