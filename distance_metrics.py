import numpy as np


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


def manhattan_distance(x1, x2):
    distance = np.sum(np.abs(x1 - x2))
    return distance
