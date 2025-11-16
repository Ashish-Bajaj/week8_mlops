import numpy as np


def poison_labels(y, percent, random_state=None):
    y = np.array(y).copy()
    if percent <= 0:
        return y
    n = len(y)
    k = int(np.round(n * percent / 100.0))
    rng = np.random.RandomState(random_state)
    indices = rng.choice(n, size=k, replace=False)
    classes = np.unique(y)
    for idx in indices:
        true = y[idx]
        other_choices = classes[classes != true]
        y[idx] = rng.choice(other_choices)
    return y
