import numpy as np
from typing import Tuple, Dict,Optional


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
    #time_for_turn_steps = max(heading_update_steps, int(time_for_turn / dt))


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


        current_azimuth = (current_azimuth + azimuth_step * dt + np.pi) % (2 * np.pi) - np.pi
        current_pitch = (current_pitch +  pitch_step * dt + np.pi) % (2 * np.pi) - np.pi

        azimuth[step_iter] = current_azimuth
        pitch[step_iter] = current_pitch
        azimuth_velocity[step_iter] = azimuth_step
        pitch_velocity[step_iter] = pitch_step


    return toroidal_to_solid_angle(np.stack([azimuth, pitch],axis=-1)), np.stack([azimuth_velocity,pitch_velocity],axis=-1), time


# def generate_bat_flight(
#     T,
#     dt = 0.1e-3,
#     boxsize = 3,
#     head_direction_turn_interval = 20e-3,
#     head_turning_std = 0.2,
#     wall_clearence = 0.2,
#     max_speed = 5,
#     rng = np.random.default_rng(seed=69),
# ) -> Dict[str,np.ndarray]:
#     n_steps = int(np.ceil(T/dt))
#     head_direction_turn_steps = int(np.ceil(head_direction_turn_interval / dt))

#     position = np.zeros((n_steps,3))
#     velocity = np.zeros((n_steps,3))
#     heading = np.zeros((n_steps, 2))
#     heading_velocity = np.zeros((n_steps,2))
#     time = np.zeros(n_steps)
#     current_position = np.ones(3) * boxsize / 2
#     current_time = 0
#     current_speed = max_speed / 2
#     current_heading = np.random.uniform(size = 3)
#     current_heading /= np.linalg.norm(current_heading)

#     position[0] = current_position
#     velocity[0] = current_heading * current_speed
#     heading[0] = np.array([
#             np.arctan2(current_heading[1], current_heading[0]),
#             np.arccos(np.clip(current_heading[2], -1.0, 1.0)),
#         ])
#     for step in range(1,n_steps):
#         if step % head_direction_turn_steps == 1:
#             # Zero out the critical components
#             upper_wall_mask = (current_position + wall_clearence) > boxsize
#             lower_wall_mask =  (current_position - wall_clearence) < 0 
#             heading_udpate = np.random.normal(size=3) * head_turning_std
#             heading_udpate[upper_wall_mask] = -head_turning_std
#             heading_udpate[lower_wall_mask] = head_turning_std

#         new_heading = current_heading + heading_udpate
#         new_heading /= np.linalg.norm(new_heading)
#         speed = max_speed
#         velocity[step] = speed * new_heading
#         position[step] = position[step - 1] + velocity[step] * dt
#         current_position = position[step]
#         heading[step] = np.array([
#             np.arctan2(new_heading[1], new_heading[0]),
#             np.arccos(np.clip(new_heading[2], -1.0, 1.0)),
#         ])
#         heading_delta = (
#             heading[step] - heading[step - 1] + np.pi
#         ) % (2 * np.pi) - np.pi
#         heading_velocity[step] = heading_delta / dt
#         current_heading = new_heading
#         current_time += dt
#     return {"pos": position, "vel": velocity, "dir": heading, "dir_vel": heading_velocity, "time": time}




def generate_bat_flight(
    T: float,
    dt: float = 0.1e-3,
    boxsize: float = 3.0,
    turn_correlation_time: float = 0.05,
    turning_std: float = 5.0,
    speed_mean: float = 3.0,
    speed_std: float = 1.5,
    speed_correlation_time: float = 0.5,
    min_speed: float = 0.3,
    max_speed: float = 7.0,
    wall_clearance: float = 0.4,
    wall_repulsion_strength: float = 3.0,
    initial_position: Optional[np.ndarray] = None,
    initial_heading: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(np.ceil(T / dt))

    position = np.zeros((n_steps, 3))
    velocity = np.zeros((n_steps, 3))
    heading = np.zeros((n_steps, 2))
    heading_velocity = np.zeros((n_steps, 2))
    time = np.arange(n_steps) * dt

    # --- initial state ---
    if initial_position is None:
        current_position = np.ones(3) * boxsize / 2
    else:
        current_position = np.array(initial_position, dtype=float).copy()

    if initial_heading is None:
        current_heading = rng.normal(size=3)
        current_heading /= np.linalg.norm(current_heading)
    else:
        current_heading = np.array(initial_heading, dtype=float)
        current_heading /= np.linalg.norm(current_heading)

    current_speed = speed_mean
    steering = np.zeros(3)  # OU "angular velocity" state driving turns

    def wall_repulsion(pos: np.ndarray) -> np.ndarray:
        """
        Soft steering push away from any wall within wall_clearance.

        Penetration is squared (not linear) so the push is gentle until
        the agent is genuinely close to a wall, and it's capped at
        `wall_repulsion_strength` per axis so it nudges the stochastic
        steering rather than overpowering it and causing deterministic
        billiard-ball-style bouncing.
        """
        lower_pen = np.clip(wall_clearance - pos, 0, None) / wall_clearance
        upper_pen = np.clip(wall_clearance - (boxsize - pos), 0, None) / wall_clearance
        return wall_repulsion_strength * (lower_pen**2 - upper_pen**2)

    def heading_to_angles(h: np.ndarray) -> np.ndarray:
        return np.array(
            [np.arctan2(h[1], h[0]), np.arccos(np.clip(h[2], -1.0, 1.0))]
        )

    ou_turn_decay = np.exp(-dt / turn_correlation_time)
    ou_turn_noise_scale = turning_std * np.sqrt(1 - ou_turn_decay**2)

    ou_speed_decay = np.exp(-dt / speed_correlation_time)
    ou_speed_noise_scale = speed_std * np.sqrt(1 - ou_speed_decay**2)

    position[0] = current_position
    velocity[0] = current_heading * current_speed
    heading[0] = heading_to_angles(current_heading)

    for step in range(1, n_steps):
        # --- speed: bounded OU process ---
        speed_noise = ou_speed_noise_scale * rng.normal()
        current_speed = (
            speed_mean + (current_speed - speed_mean) * ou_speed_decay + speed_noise
        )
        current_speed = np.clip(current_speed, min_speed, max_speed)

        # --- steering: OU process on angular velocity + wall repulsion bias ---
        turn_noise = ou_turn_noise_scale * rng.normal(size=3)
        steering = steering * ou_turn_decay + turn_noise
        steering = steering + wall_repulsion(current_position)

        # keep only the component that rotates heading (tangent to the sphere)
        tangential = steering - np.dot(steering, current_heading) * current_heading

        new_heading = current_heading + tangential * dt
        new_heading /= np.linalg.norm(new_heading)

        velocity[step] = new_heading * current_speed
        current_position = current_position + velocity[step] * dt
        current_position = np.clip(current_position, 0.0, boxsize)  # safety net

        position[step] = current_position
        heading[step] = heading_to_angles(new_heading)

        heading_delta = (heading[step] - heading[step - 1] + np.pi) % (2 * np.pi) - np.pi
        heading_velocity[step] = heading_delta / dt

        current_heading = new_heading

    return {
        "pos": position,
        "vel": velocity,
        "dir": heading,
        "dir_vel": heading_velocity,
        "time": time,
    }

