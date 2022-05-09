import argparse
import time

import numpy as np
from sklearn.metrics import precision_score
from datasets import iris
from datasets import digits
from datasets import barest_cancer
from models.gradientTreeBoosting import GradientBoosting
from models.kNeighbors import KNN
from models.logistic import Logistic

def main(args):
    print(args)
    features_name = None
    if args.datasets == 'iris':
        X_train, X_test, Y_train, Y_test, num_class = iris.get_iris_datasets()
        iris.get_visualization_iris(X=np.concatenate((X_train, X_test), axis=0),
                                    Y=np.append(Y_train, Y_test))
        features_name = iris.get_features_name()
    elif args.datasets == 'digits':
        X_train, X_test, Y_train, Y_test, num_class = digits.get_digits_datasets()
        digits.get_visualization_digits(X=np.concatenate((X_train, X_test), axis=0),
                                        Y=np.append(Y_train, Y_test))
        features_name = digits.get_features_name()
    else:
        X_train, X_test, Y_train, Y_test, num_class = barest_cancer.get_barest_cancer_datasets()
        barest_cancer.get_visualization_barest_cancer(X=np.concatenate((X_train, X_test), axis=0),
                                                      Y=np.append(Y_train, Y_test))
        features_name = barest_cancer.get_features_name()
    labels = [i for i in range(num_class)]

    if args.models == 'gtb':
        model = GradientBoosting()
    elif args.models == 'knn':
        model = KNN(num_class)
    else:
        model = Logistic()

    StartTime = time.time()

    model.fit(X_train, Y_train)
    EndTime = time.time()
    SumTime = EndTime - StartTime

    # print(precision_score(Y_train, model.predict(X_train), labels=labels, average='macro'))

    print("Train Score:{:.4}, Test Score:{:.4}, Time:{:.4}".format(precision_score(Y_train, model.predict(X_train), labels=labels, average='macro'),
                                                                   precision_score(Y_test, model.predict(X_test), labels=labels, average='macro'),
                                                                   SumTime))

    # 预测后数据分布
    if args.datasets == 'iris':
        iris.get_visualization_iris(X=np.concatenate((X_train, X_test), axis=0),
                                    Y=np.append(model.predict(X_train), model.predict(X_test)))
    elif args.datasets == 'digits':
        digits.get_visualization_digits(X=np.concatenate((X_train, X_test), axis=0),
                                        Y=np.append(model.predict(X_train), model.predict(X_test)))
    else:
        barest_cancer.get_visualization_barest_cancer(X=np.concatenate((X_train, X_test), axis=0),
                                                      Y=np.append(model.predict(X_train), model.predict(X_test)))

    # ROC曲线
    model.showROC(X_test, Y_test)

    # 混淆矩阵
    model.showConfusionMatrix(X_test, Y_test)

    # P-R曲线
    model.showPR(X_test, Y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ANN DL hw1")
    # datasets
    parser.add_argument('-d', '--datasets', type=str, default='bc',
                        choices=['bc', 'digits', 'iris'])
    # models
    parser.add_argument('-m', '--models', type=str, default='gtb',
                        choices=['gtb', 'knn', 'log'])
    args = parser.parse_args()
    main(args)