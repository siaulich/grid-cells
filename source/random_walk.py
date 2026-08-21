import numpy as np
from typing import Tuple, Dict, Optional
import tqdm


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


def toroidal_to_solid_angle(toroidal_angles: np.ndarray) -> np.ndarray:
    """
    Transform toroidal coordinates (azimuth, pitch) to spherical solid-angle
    coordinates (phi, theta) on the unit sphere.

    Parameters
    ----------
    toroidal_angles : (..., 2) ndarray
        Input angles where [..., 0] is azimuth and [..., 1] is pitch (radians),
        both defined on a torus as fully periodic variables.

    Returns
    -------
    spherical_angles : (..., 2) ndarray
        Solid-angle coordinates on the sphere:
        - phi in [-pi, pi)
        - theta in [0, pi]
    """
    toroidal_angles = np.asarray(toroidal_angles)
    if toroidal_angles.shape[-1] != 2:
        raise ValueError("toroidal_angles must have shape (..., 2)")

    azimuth = toroidal_angles[..., 0]
    pitch = toroidal_angles[..., 1]

    # Spherical azimuth
    phi = (azimuth + np.pi) % (2 * np.pi) - np.pi

    # Fold periodic pitch onto spherical elevation and convert to polar angle.
    elevation = np.arcsin(np.sin(pitch))
    theta = np.pi / 2 - elevation

    return np.stack([phi, theta], axis=-1)


def generate_head_direction_walk(
    T,
    dt=0.5e-3,
    heading_update_interval=2.5e-3,
    turn_std_azimuth=0.6 * 5,
    turn_std_pitch=0.6 * 5,
    rng=np.random.default_rng(seed=69),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a synthetic bat head-direction recording as a random walk on
    a torus. Both azimuth (phi) and pitch (theta) are fully periodic angles
    updated by independent Gaussian steps every `heading_update_interval`.

    Returns
    -------
    azimuth, pitch : (n_steps,) ndarray, ndarray
        Angles in radians, each wrapped to [-pi, pi).
    toroidal_position : (n_steps, 3) ndarray
        (x, y, z) on the torus surface (embedding of (azimuth, pitch)).
    toroidal_velocity : (n_steps, 3) ndarray
        Finite-difference derivative of toroidal_position.
    time : (n_steps,) ndarray
    """
    n_steps = int(np.ceil(T / dt))
    heading_update_steps = max(1, int(heading_update_interval / dt))
    # time_for_turn_steps = max(heading_update_steps, int(time_for_turn / dt))

    time = np.arange(n_steps) * dt
    azimuth = np.zeros(n_steps)
    pitch = np.zeros(n_steps)
    azimuth_velocity = np.zeros(n_steps)
    pitch_velocity = np.zeros(n_steps)

    current_azimuth = rng.uniform(-np.pi, np.pi)
    current_pitch = rng.uniform(-np.pi, np.pi)

    azimuth[0] = current_azimuth
    pitch[0] = current_pitch

    azimuth_step = 0
    pitch_step = 0

    for step_iter in range(1, n_steps):
        if step_iter % heading_update_steps == 0:
            azimuth_step = rng.normal(0, turn_std_azimuth)
            pitch_step = rng.normal(0, turn_std_pitch)

        current_azimuth = (current_azimuth + azimuth_step * dt + np.pi) % (
            2 * np.pi
        ) - np.pi
        current_pitch = (current_pitch + pitch_step * dt + np.pi) % (2 * np.pi) - np.pi

        azimuth[step_iter] = current_azimuth
        pitch[step_iter] = current_pitch
        azimuth_velocity[step_iter] = azimuth_step
        pitch_velocity[step_iter] = pitch_step

    return (
        toroidal_to_solid_angle(np.stack([azimuth, pitch], axis=-1)),
        np.stack([azimuth_velocity, pitch_velocity], axis=-1),
        time,
    )


def generate_bat_flight(
    T: float,
    dt: float = 0.1e-3,
    boxsize: float = 5.0,
    turning_std: float = 0.5,
    speed_mean: float = 2.5,
    speed_std: float = 1.5,
    speed_correlation_time: float = 0.5,
    max_speed: float = 5.0,
    wall_repulsion_strength: float = 1,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed=69)
    turn_clearance = boxsize / 8
    slow_clearance = boxsize / 8

    n_steps = int(np.ceil(T / dt))

    position = np.zeros((n_steps, 3))
    velocity = np.zeros((n_steps, 3))
    heading = np.zeros((n_steps, 2))
    heading_velocity = np.zeros((n_steps, 2))
    time = np.arange(n_steps) * dt

    current_position = np.ones(3) * boxsize / 2
    current_heading = rng.normal(size=3)
    current_heading /= np.linalg.norm(current_heading)

    current_speed = speed_mean
    steering = np.zeros(3)

    def wall_repulsion(pos: np.ndarray) -> np.ndarray:
        lower_pen = (turn_clearance - pos > 0).astype(float)
        upper_pen = (turn_clearance - (boxsize - pos) > 0).astype(float)
        return wall_repulsion_strength * (lower_pen - upper_pen)

    def heading_to_angles(h: np.ndarray) -> np.ndarray:
        return np.array([np.arctan2(h[1], h[0]), np.arccos(np.clip(h[2], -1.0, 1.0))])

    ou_speed_decay = np.exp(-dt / speed_correlation_time)
    ou_speed_noise_scale = speed_std * np.sqrt(1 - ou_speed_decay**2)

    position[0] = current_position
    velocity[0] = current_heading * current_speed
    heading[0] = heading_to_angles(current_heading)

    for step in tqdm.tqdm(range(1, n_steps)):
        facing_mask = (
            (current_position - turn_clearance < 0) & (current_heading < 0)
        ) | ((current_position + turn_clearance > boxsize) & (current_heading > 0))
        turn_mask = (
            (current_position - turn_clearance < 0) 
        ) | ((current_position + turn_clearance > boxsize) )


        turn_noise = turning_std * np.clip(rng.normal(size=3), -1, 1) * (~turn_mask)
        steering = steering * 0.7 + turn_noise + wall_repulsion(current_position)

        tangential = steering - np.dot(steering, current_heading) * current_heading

        new_heading = current_heading + tangential * dt
        new_heading /= np.linalg.norm(new_heading)

        slow_mask = (
            (current_position - slow_clearance < 0) & (current_heading < 0)
        ) | ((current_position + slow_clearance > boxsize) & (current_heading > 0))

        if np.any(slow_mask):
            boundary_distance = np.min(
                np.minimum(
                    current_position[slow_mask], boxsize - current_position[slow_mask]
                )
            )
            boundary_factor = np.clip(boundary_distance / slow_clearance, 0.0, 1.0)
            current_speed -= current_speed * (
                1.0 - boundary_factor
            )
        else:
            speed_noise = ou_speed_noise_scale * rng.normal()
            current_speed = (
                speed_mean + ou_speed_decay * (current_speed - speed_mean) + speed_noise
            )
        current_speed = np.clip(current_speed, 0, max_speed)

        velocity[step] = new_heading * current_speed
        current_position = current_position + velocity[step] * dt

        position[step] = current_position
        heading[step] = heading_to_angles(new_heading)

        heading_delta = (heading[step] - heading[step - 1] + np.pi) % (
            2 * np.pi
        ) - np.pi
        heading_velocity[step] = heading_delta / dt

        current_heading = new_heading

    return {
        "pos": position,
        "vel": velocity,
        "dir": heading,
        "dir_vel": heading_velocity,
        "time": time,
    }
