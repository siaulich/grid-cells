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
    rng=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a random walk trajectory in an n-dimensional box with reflective boundaries.
    
    The walk updates heading direction and speed at specified intervals, with velocity
    vectors reflected at box boundaries.
    
    Args:
        T: Total duration of the walk in seconds.
        dt: Time step in seconds. Default: 0.5e-3.
        boxsize: Size of the box in each dimension. Default: 1.0.
        heading_update_inverall: Time interval for updating heading direction in seconds. Default: 2e-2.
        speed_update_intervall: Time interval for updating speed in seconds. Default: 0.5e-1.
        ndim: Number of spatial dimensions. Default: 2.
        max_speed: Maximum speed magnitude. If None, defaults to boxsize/2.
        rng: Random number generator. If None, uses np.random.default_rng(42).
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing simulation outputs with keys:
        - 'pos': (n_steps, ndim) position trajectory
        - 'vel': (n_steps, ndim) velocity trajectory
        - 'time': (n_steps,) time array
    """
    rng = rng or np.random.default_rng(42)

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

    current_heading = rng.normal(size=ndim)
    current_heading /= np.linalg.norm(current_heading)

    time[0] = current_time
    position[0] = current_position
    velocity[0] = current_heading * current_speed

    for step_iter in tqdm.tqdm(range(1, n_steps)):
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

    return {
        "pos": position,
        "vel": velocity,
        "time": time,
    }


def compute_spherical_coordinates(heading):
    """
    Convert 3D Cartesian heading vector to spherical coordinates (azimuth, pitch).
    
    Args:
        heading: 3D heading vector or array of heading vectors.
        
    Returns:
        Array of (azimuth, pitch) angles in radians.
    """
    heading = np.asarray(heading, dtype=float)
    heading_norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    heading = heading / np.clip(heading_norm, 1e-12, None)

    azimuth = np.arctan2(heading[..., 1], heading[..., 0])
    pitch = np.arcsin(np.clip(heading[..., 2], -1.0, 1.0))

    return np.stack((azimuth, pitch), axis=-1)


def generate_bat_flight(
    T: float,
    dt: float = 0.1e-3,
    boxsize: float = 5.0,
    turning_std: float = 0.5,
    speed_mean: float = 2.5,
    speed_std: float = 1.5,
    speed_correlation_time: float = 0.5,
    min_speed: float = 1.2,
    max_speed: float = 5.0,
    wall_repulsion_strength: float = 5,
    rng: Optional[np.random.Generator] = None,
    initial_position=None,
    initial_speed=None,
    initial_heading=None,
) -> Dict[str, np.ndarray]:
    """
    Generate a bat flight trajectory in a 3D box with realistic dynamics.
    
    Parameters
    ----------
    T : float
        Total simulation time in seconds.
    dt : float, optional
        Time step for integration. Default is 0.1 ms.
    boxsize : float, optional
        Size of the cubic domain [0, boxsize]^3. Default is 5.0.
    turning_std : float, optional
        Standard deviation of turning angles in degrees. Default is 0.5.
    speed_mean : float, optional
        Mean speed in m/s. Default is 2.5.
    speed_std : float, optional
        Standard deviation of speed. Default is 1.5.
    speed_correlation_time : float, optional
        Correlation time for speed fluctuations in seconds. Default is 0.5.
    min_speed : float, optional
        Minimum speed in m/s. Default is 1.2.
    max_speed : float, optional
        Maximum speed in m/s. Default is 5.0.
    wall_repulsion_strength : float, optional
        Strength of wall repulsion force. Default is 5.
    rng : np.random.Generator, optional
        Random number generator. If None, uses default with seed 42.
    initial_position : array-like, optional
        Initial position [x, y, z]. If None, starts at box center.
    initial_speed : float, optional
        Initial speed. If None, uses speed_mean.
    initial_heading : array-like, optional
        Initial heading vector [hx, hy, hz]. If None, random unit vector.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing simulation outputs with keys:
        - 'pos': (n_steps, 3) position trajectory
        - 'vel': (n_steps, 3) velocity trajectory
        - 'dir_torus': (n_steps, 2) heading in toroid coordinates
        - 'dir_sphere': (n_steps, 2) heading in spherical coordinates
        - 'dir_vel': (n_steps, 2) heading velocity
        - 'time': (n_steps,) time array
    """
    
    rng = rng or np.random.default_rng(42)

    turn_clearance = boxsize / 8
    slow_clearance = boxsize / 8
    n_steps = int(np.ceil(T / dt))

    # Arrays to store state
    position = np.zeros((n_steps, 3))
    velocity = np.zeros((n_steps, 3))
    toroid_heading = np.zeros((n_steps, 2))
    sphere_heading = np.zeros((n_steps, 2))
    heading_velocity = np.zeros((n_steps, 2))
    time = np.arange(n_steps) * dt

    # Initialize state
    if initial_position is None:
        current_position = np.ones(3) * boxsize / 2
    else:
        initial_position = np.array(initial_position).flatten()
        if initial_position.shape == (3,):
            current_position = initial_position
        else:
            raise ValueError("initial_position muts be a (3,) array")

    # Initialize state
    if initial_heading is None:
        current_heading = rng.normal(size=3)
        current_heading /= np.linalg.norm(current_heading)
    else:
        initial_heading = np.array(initial_heading).flatten()
        if initial_heading.shape == (3,):
            initial_heading /= np.linalg.norm(initial_heading)
            current_heading = initial_heading
        else:
            raise ValueError("initial_heading must be a (3,) array")

    # Initialize state
    if initial_speed is None:
        current_speed = speed_mean
    else:
        if isinstance(initial_speed, float):
            current_speed = initial_speed
        else:
            raise ValueError("initial_speed must be a float")

    # OU parameters
    ou_speed_decay = np.exp(-dt / speed_correlation_time)
    ou_speed_noise_scale = speed_std * np.sqrt(1 - ou_speed_decay**2)

    steering_correlation_time = 0.2
    ou_steer_decay = np.exp(-dt / steering_correlation_time)
    ou_steer_noise_scale = turning_std * np.sqrt(1 - ou_steer_decay**2)

    def wall_repulsion(pos: np.ndarray) -> np.ndarray:
        eps = 1e-3
        dist_lower = np.clip(pos, eps, turn_clearance)
        dist_upper = np.clip(boxsize - pos, eps, turn_clearance)

        force_lower = (1.0 / dist_lower) - (1.0 / turn_clearance)
        force_upper = (1.0 / dist_upper) - (1.0 / turn_clearance)

        return wall_repulsion_strength * (force_lower - force_upper)

    current_azimuth = 0
    current_pitch = 0

    # Steering is now natively [d_azimuth, d_pitch]
    steering_angles = np.zeros(2)

    position[0] = current_position
    velocity[0] = current_speed * current_heading
    toroid_heading[0] = np.array([current_azimuth, current_pitch])
    sphere_heading[0] = compute_spherical_coordinates(current_heading)

    # OU params for steering
    steering_correlation_time = 0.2
    ou_steer_decay = np.exp(-dt / steering_correlation_time)
    ou_steer_noise_scale = turning_std * np.sqrt(1 - ou_steer_decay**2)

    max_angular_vel = 20.0  # rad/s

    for step in tqdm.tqdm(range(1, n_steps)):
        h_x = np.cos(current_pitch) * np.cos(current_azimuth)
        h_y = np.cos(current_pitch) * np.sin(current_azimuth)
        h_z = np.sin(current_pitch)
        current_heading = np.array([h_x, h_y, h_z])

        facing_mask = (
            (current_position - turn_clearance < 0) & (current_heading < 0)
        ) | ((current_position + turn_clearance > boxsize) & (current_heading > 0))
        repulsion_force = wall_repulsion(current_position)

        u_azimuth = np.array([-np.sin(current_azimuth), np.cos(current_azimuth), 0.0])
        u_pitch = np.array(
            [
                -np.cos(current_azimuth) * np.sin(current_pitch),
                -np.sin(current_azimuth) * np.sin(current_pitch),
                np.cos(current_pitch),
            ]
        )

        repulsion_angles = np.array(
            [np.dot(repulsion_force, u_azimuth), np.dot(repulsion_force, u_pitch)]
        )

        turn_noise = ou_steer_noise_scale * rng.normal(size=2)
        if np.any(facing_mask):
            turn_noise = np.zeros(2)

        steering_angles = steering_angles * ou_steer_decay + turn_noise
        total_steering = steering_angles + repulsion_angles

        steering_norm = np.linalg.norm(total_steering)
        if steering_norm > max_angular_vel:
            total_steering = total_steering * (max_angular_vel / steering_norm)

        current_azimuth = (current_azimuth + total_steering[0] * dt) % (2 * np.pi)
        current_pitch = (current_pitch + total_steering[1] * dt) % (2 * np.pi)

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
            braking_rate = 5.0
            current_speed -= braking_rate * (1.0 - boundary_factor) * current_speed * dt
            current_speed = np.clip(current_speed, 0.0, max_speed)
        else:
            speed_noise = ou_speed_noise_scale * rng.normal()
            current_speed = (
                speed_mean + ou_speed_decay * (current_speed - speed_mean) + speed_noise
            )
            current_speed = np.clip(current_speed, min_speed, max_speed)

        velocity[step] = current_heading * current_speed
        current_position = current_position + velocity[step] * dt
        # current_position = np.clip(current_position, 1e-4, boxsize - 1e-4)

        position[step] = current_position
        toroid_heading[step] = [current_azimuth, current_pitch]
        heading_velocity[step] = total_steering
        sphere_heading[step] = compute_spherical_coordinates(current_heading)

    return {
        "pos": position,
        "vel": velocity,
        "dir_torus": toroid_heading,
        "dir_sphere": sphere_heading,
        "dir_vel": heading_velocity,
        "time": time,
    }


from typing import Dict, Optional
import numpy as np
import tqdm


def generate_2d_bat_flight(
    T: float,
    dt: float = 0.1e-3,
    boxsize: float = 5.0,
    turning_std: float = 0.5,
    speed_mean: float = 2.5,
    speed_std: float = 1.5,
    speed_correlation_time: float = 0.5,
    min_speed: float = 1.2,
    max_speed: float = 5.0,
    wall_repulsion_strength: float = 5,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate 2D bat flight in a bounded cubic box with realistic movement dynamics.
    
    The trajectory is constrained to a 2D horizontal plane (z = boxsize/2) with 
    Ornstein-Uhlenbeck processes for speed and steering. Wall repulsion and speed 
    modulation near boundaries provide realistic boundary avoidance.
    
    Args:
        T: Total simulation time in seconds.
        dt: Time step in seconds.
        boxsize: Size of cubic domain [0, boxsize]^3.
        turning_std: Standard deviation for steering noise (rad/s).
        speed_mean: Mean speed (m/s).
        speed_std: Standard deviation of speed (m/s).
        speed_correlation_time: Correlation time for speed OU process (s).
        min_speed: Minimum speed constraint (m/s).
        max_speed: Maximum speed constraint (m/s).
        wall_repulsion_strength: Strength of repulsive force from walls.
        rng: Random number generator. Defaults to np.random.default_rng(42).
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing simulation outputs with keys:
        - 'pos': (n_steps, 3) position trajectory
        - 'vel': (n_steps, 3) velocity trajectory
        - 'dir_torus': (n_steps, 2) heading in toroid coordinates
        - 'dir_sphere': (n_steps, 2) heading in spherical coordinates
        - 'dir_vel': (n_steps, 2) heading velocity
        - 'time': (n_steps,) time array
    """

    rng = rng or np.random.default_rng(42)

    turn_clearance = boxsize / 8
    slow_clearance = boxsize / 8
    n_steps = int(np.ceil(T / dt))

    # Arrays retained in 3D dimensions to match your original output structure
    position = np.zeros((n_steps, 3))
    velocity = np.zeros((n_steps, 3))
    toroid_heading = np.zeros((n_steps, 2))
    sphere_heading = np.zeros((n_steps, 3))
    heading_velocity = np.zeros((n_steps, 2))
    time = np.arange(n_steps) * dt

    # Initialize state (Fixing z at boxsize / 2 for a flat horizontal plane flight)
    current_position = np.array([boxsize / 2, boxsize / 2, boxsize / 2])
    current_azimuth = rng.uniform(0, 2 * np.pi)
    current_pitch = 0.0  # Locked to zero for a pure plane
    current_speed = speed_mean

    # OU parameters for speed
    ou_speed_decay = np.exp(-dt / speed_correlation_time)
    ou_speed_noise_scale = speed_std * np.sqrt(1 - ou_speed_decay**2)

    # OU parameters for steering
    steering_correlation_time = 0.2
    ou_steer_decay = np.exp(-dt / steering_correlation_time)
    ou_steer_noise_scale = turning_std * np.sqrt(1 - ou_steer_decay**2)

    max_angular_vel = 20.0  # rad/s
    steering_angles = np.zeros(2)

    def wall_repulsion(pos: np.ndarray) -> np.ndarray:
        eps = 1e-3
        # For the z-dimension, we can ignore or minimize repulsion if we lock it,
        # but let's calculate it normally across all 3 axes to keep logic uniform
        dist_lower = np.clip(pos, eps, turn_clearance)
        dist_upper = np.clip(boxsize - pos, eps, turn_clearance)

        force_lower = (1.0 / dist_lower) - (1.0 / turn_clearance)
        force_upper = (1.0 / dist_upper) - (1.0 / turn_clearance)

        return wall_repulsion_strength * (force_lower - force_upper)

    # Initial mapping
    h_x = np.cos(current_pitch) * np.cos(current_azimuth)
    h_y = np.cos(current_pitch) * np.sin(current_azimuth)
    h_z = np.sin(current_pitch)
    current_heading = np.array([h_x, h_y, h_z])

    position[0] = current_position
    velocity[0] = current_speed * current_heading
    toroid_heading[0] = np.array([current_azimuth, current_pitch])
    sphere_heading[0] = (
        current_heading  # assuming compute_spherical_coordinates normalizes/converts to unit vector
    )

    for step in tqdm.tqdm(range(1, n_steps)):
        h_x = np.cos(current_pitch) * np.cos(current_azimuth)
        h_y = np.cos(current_pitch) * np.sin(current_azimuth)
        h_z = np.sin(current_pitch)
        current_heading = np.array([h_x, h_y, h_z])

        # We only check facing bounds for x and y to prevent wall crashes in the plane
        facing_mask_xy = (
            (current_position[:2] - turn_clearance < 0) & (current_heading[:2] < 0)
        ) | (
            (current_position[:2] + turn_clearance > boxsize)
            & (current_heading[:2] > 0)
        )

        repulsion_force = wall_repulsion(current_position)

        u_azimuth = np.array([-np.sin(current_azimuth), np.cos(current_azimuth), 0.0])
        u_pitch = np.array(
            [
                -np.cos(current_azimuth) * np.sin(current_pitch),
                -np.sin(current_azimuth) * np.sin(current_pitch),
                np.cos(current_pitch),
            ]
        )

        repulsion_angles = np.array(
            [np.dot(repulsion_force, u_azimuth), np.dot(repulsion_force, u_pitch)]
        )

        turn_noise = ou_steer_noise_scale * rng.normal(size=2)
        # Lock pitch noise to 0 so it never tries to drift out of the plane
        turn_noise[1] = 0.0

        if np.any(facing_mask_xy):
            turn_noise[0] = 0.0

        steering_angles = steering_angles * ou_steer_decay + turn_noise
        total_steering = steering_angles + repulsion_angles
        total_steering[1] = 0.0  # Force pitch steering velocity to zero

        steering_norm = np.linalg.norm(total_steering)
        if steering_norm > max_angular_vel:
            total_steering = total_steering * (max_angular_vel / steering_norm)

        current_azimuth = (current_azimuth + total_steering[0] * dt) % (2 * np.pi)
        current_pitch = 0.0  # Keep pitch locked rigidly at 0

        slow_mask_xy = (
            (current_position[:2] - slow_clearance < 0) & (current_heading[:2] < 0)
        ) | (
            (current_position[:2] + slow_clearance > boxsize)
            & (current_heading[:2] > 0)
        )

        if np.any(slow_mask_xy):
            boundary_distance = np.min(
                np.minimum(
                    current_position[:2][slow_mask_xy],
                    boxsize - current_position[:2][slow_mask_xy],
                )
            )
            boundary_factor = np.clip(boundary_distance / slow_clearance, 0.0, 1.0)
            braking_rate = 5.0
            current_speed -= braking_rate * (1.0 - boundary_factor) * current_speed * dt
            current_speed = np.clip(current_speed, 0.0, max_speed)
        else:
            speed_noise = ou_speed_noise_scale * rng.normal()
            current_speed = (
                speed_mean + ou_speed_decay * (current_speed - speed_mean) + speed_noise
            )
            current_speed = np.clip(current_speed, min_speed, max_speed)

        velocity[step] = current_heading * current_speed
        current_position = current_position + velocity[step] * dt

        # Explicitly pin the z-coordinate to the center plane
        current_position[2] = boxsize / 2

        position[step] = current_position
        toroid_heading[step] = [current_azimuth, current_pitch]
        heading_velocity[step] = total_steering
        sphere_heading[step] = current_heading

    return {
        "pos": position,
        "vel": velocity,
        "dir_torus": toroid_heading,
        "dir_sphere": sphere_heading,
        "dir_vel": heading_velocity,
        "time": time,
    }
