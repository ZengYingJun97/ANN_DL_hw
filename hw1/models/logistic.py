from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import scikitplot as skplt
import matplotlib.pyplot as plt

class Logistic:
    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                 verbose=0, warm_start=False, n_jobs=1):
        self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,
                                        fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                                        random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class,
                                        verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

        self.param_dict = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
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
                       title="Digits Precision-Recall Curve", figsize=(12,6))
        plt.show()
