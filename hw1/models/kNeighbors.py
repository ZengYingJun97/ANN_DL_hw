from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import scikitplot as skplt
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                 leaf_size=30, p=2, metric='minkowski', metric_params=None,
                 n_jobs=1):
        self.model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                                    leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params,
                                                    n_jobs=n_jobs)
        n_range = range(1, 31)
        weight_options = ['uniform', 'distance']
        p = [1, 2]
        self.param_dict = {"n_neighbors": n_range, "weights": weight_options, "p": p}
        self.estimator = GridSearchCV(self.model, param_grid=self.param_dict, cv=10)

    def fit(self, X, Y):
        self.estimator.fit(X, Y)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, Y):
        return self.estimator.score(X, Y)

    def showFeatureImportance(self, features_name):
        skplt.estimators.plot_feature_importances(self.model,
                                                  feature_names=features_name)
        plt.show()

    def showROC(self, X, Y):
        skplt.metrics.plot_roc_curve(Y, self.predict_proba(X),
                                     title="Digits ROC Curve", figsize=(12, 6))
        plt.show()

    def showConfusionMatrix(self, X, Y):
        skplt.metrics.plot_confusion_matrix(Y, self.predict(X),
                                            normalize=True,
                                            title="Confusion Matrix",
                                            cmap="Oranges")

        plt.show()

    def showPR(self, X, Y):
        skplt.metrics.plot_precision_recall_curve(Y, self.predict_proba(X),
                                                  title="Digits Precision-Recall Curve", figsize=(12, 6))
        plt.show()
