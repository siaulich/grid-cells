import numpy as np
from typing import Union, Tuple, List


def activity_map(
    position: np.ndarray,
    activity: np.ndarray,
    nbins: Union[Tuple[int, ...], int] = 50,
) -> Tuple[np.ndarray, List[np.ndarray]]:

    ndim = position.shape[1]
    if isinstance(nbins, int):
        nbins = (nbins,) * ndim

    pos_min = position.min(axis=0)
    pos_max = position.max(axis=0)
    edges = [np.linspace(pos_min[i], pos_max[i], nbins[i] + 1) for i in range(ndim)]

    activity_sum, edges = np.histogramdd(position, bins=edges, weights=activity)
    counts, _ = np.histogramdd(position, bins=edges)

    mean_activity = np.divide(
        activity_sum, counts,
        out=np.zeros_like(activity_sum),
        where=counts > 0,
    )

    return mean_activity, edges