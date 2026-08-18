import numpy as np
from typing import Union, Tuple, List


def activity_map(
    position: np.ndarray,
    activity: np.ndarray,
    nbins: Union[Tuple[int, ...], int] = 50,
) -> Tuple[np.ndarray, List[np.ndarray]]:

    if position.ndim == 2:
        ndim = position.shape[1]
    elif position.ndim == 1:
        ndim = 1
    else:
        raise ValueError("WTF?")
    if isinstance(nbins, int):
        nbins = (nbins,) * ndim

    pos_min = position.min(axis=0)
    pos_max = position.max(axis=0)
    if ndim > 1:
        edges = [np.linspace(pos_min[i], pos_max[i], nbins[i] + 1) for i in range(ndim)]
    else:
        edges = [np.linspace(pos_min, pos_max, nbins[0] + 1)]

    activity_sum, edges = np.histogramdd(position, bins=edges, weights=activity)
    counts, _ = np.histogramdd(position, bins=edges)

    mean_activity = np.divide(
        activity_sum, counts,
        out=np.zeros_like(activity_sum),
        where=counts > 0,
    )

    return mean_activity, edges