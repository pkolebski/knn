from pathlib import Path

import pandas as pd


def load_dataset(path: Path, separator: str = ','):
    dataset = pd.read_csv(path, sep=separator)
    y = dataset.Target
    x = dataset.drop(columns=['Target'])
    return x.to_numpy(), y.to_numpy()
