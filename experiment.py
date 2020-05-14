from typing import Callable, Tuple

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from data import DATASETS_PATH
from data.load_dataset import load_dataset
from knn import KNN
from distance_metrics import euclidean_distance, manhattan_distance
from voting_methods import distance_weighted_voting, majority_voting, gaussian_weighted_voting

sns.set()
iris_x, iris_y = load_dataset(DATASETS_PATH / 'iris' / 'iris.data')
iris_dataset = ('Iris Dataset', iris_x, iris_y)
glass_x, glass_y = load_dataset(DATASETS_PATH / 'glass' / 'glass.data')
glass_dataset = ('Glass Dataset', glass_x, glass_y)
wine_x, wine_y = load_dataset(DATASETS_PATH / 'wine' / 'wine.data')
wine_dataset = ('WWine Dataset', wine_x, wine_y)
seeds_x, seeds_y = load_dataset(DATASETS_PATH / 'seeds' / 'seeds_dataset.csv', separator='\t')
seeds_dataset = ('Seeds dataset', seeds_x, seeds_y)


def run_experiment(x: np.ndarray, y: np.ndarray, voting_method: Callable, distance_metric: Callable,
                   max_k: int) -> list:
    results = list()
    for k in tqdm(range(1, max_k)):
        knn = KNN(
            voting_method=voting_method,
            distance_metric=distance_metric,
            n_neighbors=k,
            preprocessing=StandardScaler(),
        )
        results.append(np.mean(cross_val_score(knn, x, y, cv=10)))
    return results


def run_experiments_for_dataset(dataset: Tuple[str, np.ndarray, np.ndarray], max_k: int = 20):
    dataset_name, x, y = dataset
    results = []
    dist_methods = [
        ('Manhattan distance', manhattan_distance),
        ('Euclidean distance', euclidean_distance),
    ]
    voting_methods = [
        ('Majority voting', majority_voting),
        ('Distance weighted voting', distance_weighted_voting),
        ('Gaussian weighted voting', gaussian_weighted_voting)
    ]
    for dist_name, dist in dist_methods:
        for voting_name, voting in voting_methods:
            name = f'{dist_name} and {voting_name}'
            fscores = run_experiment(x, y, voting, dist, max_k)
            results.append((name, fscores))
    fig = plt.figure()
    for line_name, fscores in results:
        sns.lineplot(list(range(1, max_k)), fscores, label=line_name)
    plt.title(dataset_name)

    # plt.yticks(list(range(int(min(6, 10 * min(fscores) - 0.1)), 10, 1)))
    plt.legend(bbox_to_anchor=(1, 0), loc="lower right",
               bbox_transform=fig.transFigure, ncol=3)
    plt.tight_layout()
    plt.show()


run_experiments_for_dataset(iris_dataset)
run_experiments_for_dataset(wine_dataset)
run_experiments_for_dataset(glass_dataset)
run_experiments_for_dataset(seeds_dataset)
# # sns.legend()
# plt.show()
