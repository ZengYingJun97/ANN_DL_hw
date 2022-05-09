from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import scikitplot as skplt
import matplotlib.pyplot as plt

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, subsample=1.0, loss='deviance', max_depth=4, random_state=1):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                                subsample=subsample, loss=loss, max_depth=max_depth, random_state=random_state)

        n_estimators = [100]
        # learning_rate = [0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        # subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        loss = ['deviance', 'exponential']
        self.param_dict = {'n_estimators': n_estimators, 'loss': loss}
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