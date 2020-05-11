from pathlib import Path

import pandas as pd


def load_dataset(path: Path):
    dataset = pd.read_csv(path)
    y = dataset.Target
    x = dataset.drop(columns=['Target'])
    return x.to_numpy(), y.to_numpy()
