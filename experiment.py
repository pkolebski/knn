import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from data import DATASETS_PATH
from data.load_dataset import load_dataset
from knn import KNN
from distance_metrics import euclidean_distance
from voting_methods import distance_weighted_voting


x, y = load_dataset(DATASETS_PATH / 'iris' / 'iris.data')

knn = KNN(
    voting_method=distance_weighted_voting,
    distance_metric=euclidean_distance,
    n_neighbors=5,
    preprocessing=StandardScaler(),
)

print(np.mean(cross_val_score(knn, x, y, cv=10)))

