import numpy as np
from typing import Tuple


def generate_random_walk(
    T,
    dt=0.5e-3,
    boxsize=1.0,
    heading_update_inverall=2e-2,
    speed_update_intervall=0.5e-1,
    ndim=2,
    max_speed=None,
    rng=np.random.default_rng(seed=69),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a systematic, space-filling random walk in a box [0, boxsize]^ndim.

    The trajectory visits the box on a regular grid using a snake ordering so that
    the walk covers the full domain in an organized way rather than wandering in a
    purely diffusive fashion. The returned arrays keep the same interface as the
    original implementation: position, velocity, time.
    """

    n_steps = int(np.ceil(T / dt))
    heading_update_steps = int(heading_update_inverall / dt)
    speed_update_steps = int(speed_update_intervall / dt)
    time = np.zeros(n_steps)
    position = np.zeros((n_steps, ndim))
    velocity = np.zeros((n_steps, ndim))

    if max_speed is None:
        max_speed = boxsize / 2

    turn_std = 0.6

    current_position = np.ones(ndim) * boxsize / 2
    current_time = 0
    current_speed = rng.uniform(max_speed / 2, max_speed)

    # Random unit vector in ndim dimensions (generalizes the 2D angle)
    current_heading = rng.normal(size=ndim)
    current_heading /= np.linalg.norm(current_heading)

    time[0] = current_time
    position[0] = current_position
    velocity[0] = current_heading * current_speed

    for step_iter in range(1, n_steps):
        if step_iter % speed_update_steps == 0:
            current_speed = rng.uniform(max_speed / 2, max_speed)

        if step_iter % heading_update_steps == 0:
            current_heading = current_heading + rng.normal(0, turn_std, size=ndim)
            current_heading /= np.linalg.norm(current_heading)

        current_velocity = current_heading * current_speed
        proposed_position = current_position + current_velocity * dt

        below = proposed_position < 0
        above = proposed_position > boxsize
        flip = below | above
        if np.any(flip):
            current_velocity[flip] = -current_velocity[flip]
            current_heading[flip] = -current_heading[flip]
            proposed_position = current_position + current_velocity * dt

        current_position = proposed_position
        current_time += dt

        time[step_iter] = current_time
        position[step_iter] = current_position
        velocity[step_iter] = current_velocity

    return position, velocity, time

def generate_systematic_walk(
    T,
    dt=0.5e-3,
    boxsize=1.0,
    line_spacing=0.05,
    speed=None,
    ndim=2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a systematic, space-filling trajectory in a box [0, boxsize]^ndim.

    The animal moves along a boustrophedon ("lawnmower"/snake) path that visits
    a regular grid of points spaced `line_spacing` apart, sweeping back and forth
    along the fastest axis and stepping to the next line/plane once a sweep is
    complete. In 3D this generalizes to a snake-of-snakes: each z-plane is fully
    raster-scanned in x/y before moving to the next z-plane. This guarantees full,
    even coverage of the box, which is what you want when mapping out grid-cell
    firing fields. Interface (position, velocity, time) matches the random-walk
    version.

    Parameters
    ----------
    T : float
        Total duration of the trajectory (s).
    dt : float
        Sampling time step (s).
    boxsize : float
        Side length of the (hyper)cube environment.
    line_spacing : float
        Spacing between adjacent sweep lines (and, in 3D, between planes).
        Smaller values cover the box more densely but the grid of waypoints
        (and memory) grows as (boxsize/line_spacing)**ndim, so keep this
        modest in 3D.
    speed : float, optional
        Running speed (units/s) along the path. Defaults to a value that
        completes ~4 full sweeps of the box over the course of T.
    ndim : int
        Number of spatial dimensions (works for any ndim >= 1, typically 2 or 3).
    rng : np.random.Generator
        Unused for the deterministic path itself; kept for interface parity
        with generate_random_walk (e.g. if you want to add a random phase
        offset or jitter on top).

    Returns
    -------
    position, velocity, time : np.ndarray
    """

    n_steps = int(np.ceil(T / dt))
    time = np.arange(n_steps) * dt

    # --- build the waypoints of a boustrophedon path over a regular grid ---
    n_per_dim = max(2, int(np.ceil(boxsize / line_spacing)) + 1)
    shape = tuple(n_per_dim for _ in range(ndim))

    def snake_indices(shape):
        # Recursively build grid indices in continuous snake ("lawnmower") order.
        # The last axis is the slowest-varying one; each time it advances, the
        # order of the inner (faster) axes is reversed so the path stays connected.
        if len(shape) == 1:
            return np.arange(shape[0])[:, None]
        sub = snake_indices(shape[:-1])
        n = shape[-1]
        blocks = []
        for i in range(n):
            block = sub if i % 2 == 0 else sub[::-1]
            col = np.full((block.shape[0], 1), i)
            blocks.append(np.hstack([block, col]))
        return np.vstack(blocks)

    idx = snake_indices(shape)                    # (M, ndim) grid indices, snake order
    axis_coords = np.linspace(0, boxsize, n_per_dim)
    waypoints = axis_coords[idx]                   # (M, ndim) actual coordinates

    # --- arc-length parametrisation of the path ---
    seg_vecs = np.diff(waypoints, axis=0)                  # (M-1, ndim)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)             # (M-1,)
    nonzero = seg_lens > 0
    seg_dirs = np.zeros_like(seg_vecs)
    seg_dirs[nonzero] = seg_vecs[nonzero] / seg_lens[nonzero, None]

    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    L_total = cum_len[-1]

    if speed is None:
        speed = 4 * L_total / T  # ~4 full coverage sweeps over the trajectory

    # --- sample the path at every time step (loops once L_total is covered) ---
    s = (speed * time) % L_total
    seg_idx = np.clip(np.searchsorted(cum_len, s, side="right") - 1, 0, len(seg_lens) - 1)
    frac = np.zeros_like(s)
    nz = seg_lens[seg_idx] > 0
    frac[nz] = (s[nz] - cum_len[seg_idx][nz]) / seg_lens[seg_idx][nz]

    position = waypoints[seg_idx] + frac[:, None] * seg_vecs[seg_idx]
    velocity = seg_dirs[seg_idx] * speed

    return position, velocity, time