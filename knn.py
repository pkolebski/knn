import numpy as np
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator


class KNN(BaseEstimator):
    def __init__(self, voting_method, distance_metric, n_neighbors: int = 5, preprocessing=None):
        self.voting_method = voting_method
        self.distance_metric = distance_metric
        self.x = None
        self.y = None
        self.class_to_int = None
        self.int_to_class = None
        self.n_neighbors = n_neighbors
        self.preprocessing = preprocessing

    def fit(self, x: np.ndarray, y: np.ndarray):
        classes = set(y)
        self.class_to_int = {cls: i for i, cls in enumerate(classes)}
        self.int_to_class = {i: cls for i, cls in enumerate(classes)}
        self.y = np.vectorize(self.class_to_int.get)(y)
        self.x = x
        if self.preprocessing is not None:
            self.x = self.preprocessing.fit_transform(self.x)
        return self

    def score(self, x, y):
        predicted = self.predict(x)
        return f1_score(y, predicted, average='micro')

    def predict(self, x: np.ndarray):
        if self.preprocessing is not None:
            x = self.preprocessing.transform(x)
        distances = np.empty((self.x.shape[0], x.shape[0]), dtype=tuple)
        for i, (x_train, y_train) in enumerate(zip(self.x, self.y)):
            for j, x_test in enumerate(x):
                distances[i, j] = (self.distance_metric(x_train, x_test), y_train)
        predicted = []
        for i, x_test in enumerate(x):
            predicted.append(self.voting_method(distances[:, i], self.n_neighbors))
        return np.vectorize(self.int_to_class.get)(predicted)
