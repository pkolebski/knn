from collections import defaultdict

import numpy as np


def majority_voting(distances: np.ndarray, k: int) -> int:
    distances = sorted(distances, key=lambda x: x[0])[:k]
    votes = defaultdict(float)
    for dist, cls in distances:
        votes[cls] += 1
    return get_prediction(votes)


def distance_weighted_voting(distances: np.ndarray, k: int, epsilon: float = 1e-3) -> int:
    distances = sorted(distances, key=lambda x: x[0])[:k]
    votes = defaultdict(float)
    for dist, cls in distances:
        votes[cls] += 1 / (dist + epsilon)
    return get_prediction(votes)


def gaussian_weighted_voting(distances: np.ndarray, k: int) -> int:
    distances = sorted(distances, key=lambda x: x[0])[:k]
    votes = defaultdict(float)
    for dist, cls in distances:
        votes[cls] += 1 / np.sqrt(2 * np.pi) * np.exp(-(dist ** 2) / 2)
    return get_prediction(votes)


def get_prediction(votes: dict) -> int:
    return max(votes.keys(), key=lambda key: votes[key])
