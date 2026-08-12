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
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: position, velocity, time
    """

    n_steps = int(np.ceil(T / dt))
    heading_update_steps = int(heading_update_inverall / dt)
    speed_update_steps = int(speed_update_intervall / dt)
    time = np.zeros(n_steps)
    position = np.zeros((n_steps, ndim))
    velocity = np.zeros((n_steps, ndim))

    if max_speed is None:
        max_speed = boxsize / 2
    if ndim != 2:
        raise NotImplementedError("3-dimensional Random Walks are not yet implemented")
    turn_std = 0.6
    current_position = np.ones(ndim) * boxsize / 2
    current_time = 0
    current_speed = rng.uniform(max_speed / 2, max_speed)
    current_heading = rng.uniform(0, 2 * np.pi)
    time[0] = current_time
    position[0] = current_position
    velocity[0] = np.array(
        [
            np.cos(current_heading) * current_speed,
            np.sin(current_heading) * current_speed,
        ]
    )
    for step_iter in range(1, n_steps):
        if step_iter % speed_update_steps == 0:
            current_speed = rng.uniform(max_speed / 2, max_speed)
        if step_iter % heading_update_steps == 0:
            current_heading += rng.normal(0, turn_std)

        vx, vy = (
            np.cos(current_heading) * current_speed,
            np.sin(current_heading) * current_speed,
        )
        proposed_position = current_position + np.array([vx, vy]) * dt
        if proposed_position[0] < 0:
            vx = -vx
            current_heading = np.pi - current_heading
        elif proposed_position[0] > boxsize:
            vx = -vx
            current_heading = np.pi - current_heading
        if proposed_position[1] < 0:
            vy = -vy
            current_heading = -current_heading
        elif proposed_position[1] > boxsize:
            vy = -vy
            current_heading = -current_heading

        current_position = current_position + np.array([vx, vy]) * dt

        current_time += dt
        time[step_iter] = current_time
        position[step_iter] = current_position
        velocity[step_iter] = np.array([vx, vy])
    return position, velocity, time
