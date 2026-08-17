import numpy as np
import matplotlib.pyplot as plt
import tqdm
from source import generate_random_walk, activity_map, generate_systematic_walk
import itertools

ndim = 3
n = 32

tau = 10e-3
dt = 0.5e-3

lambda_net = 12
beta = 3.0 / lambda_net**2
gamma = 1.05 * beta
a_weight = 1

l_shift = 2.0
alpha = 0.10315

B0 = 1
seed = 0

def periodic_kernel(n, ndim, a_weight, gamma, beta, l_shift=0.0, e_theta=None):
    """
    ndim-dimensional generalization of the periodic (wrapped) difference-
    of-Gaussians kernel, shifted by l_shift along direction e_theta.
    """
    if e_theta is None:
        e_theta = np.zeros(ndim)

    idx = np.arange(n)
    d = idx - n // 2
    d = np.where(d > n / 2, d - n, d)
    d = np.where(d < -n / 2, d + n, d)

    # ndim coordinate grids, each of shape (n,)*ndim
    grids = np.meshgrid(*([d] * ndim), indexing="ij")

    r2 = np.zeros((n,) * ndim)
    for axis, g in enumerate(grids):
        s = g - l_shift * e_theta[axis]
        r2 = r2 + s**2

    K = a_weight * np.exp(-gamma * r2) - np.exp(-beta * r2)
    K = np.fft.ifftshift(K)
    return K

dir_keys = list(itertools.product([1, -1], repeat=ndim))
dir_vectors = {}
for key in dir_keys:
    v = np.array(key, dtype=float)
    v /= np.linalg.norm(v)
    dir_vectors[key] = v

block = np.empty((2,) * ndim, dtype=object)
for pos in itertools.product([0, 1], repeat=ndim):
    key_index = pos[0]
    for axis in range(1, ndim):
        key_index = key_index * 2 + pos[axis]
    block[pos] = dir_keys[key_index]

reps = (n // 2,) * ndim
theta_dir = np.tile(block, reps)

directed_kernels = {}
directed_masks = {}

for key in dir_keys:
    directed_kernels[key] = np.fft.fftn(
        periodic_kernel(
            n,
            ndim,
            a_weight,
            gamma,
            beta,
            l_shift=l_shift,
            e_theta=dir_vectors[key],
        )
    )
    directed_masks[key] = np.vectorize(lambda e, key=key: e == key)(theta_dir)

def recurrent_input(s):
    rec_input = np.zeros_like(s)
    for key in dir_keys:
        rec_input += np.real(
            np.fft.ifftn(np.fft.fftn(directed_masks[key] * s) * directed_kernels[key])
        )
    return rec_input

e_theta_components = [
    np.vectorize(lambda key, axis=axis: dir_vectors[key][axis])(theta_dir)
    for axis in range(ndim)
]

def feedforward_input(v):
    drive = np.zeros_like(theta_dir, dtype=float)
    for axis in range(ndim):
        drive = drive + e_theta_components[axis] * v[axis]
    return B0 * (1.0 + alpha * drive)

def step(s, v):
    total_input = recurrent_input(s) + feedforward_input(v)
    return s + (dt / tau) * (-s + np.maximum(total_input, 0.0))


## Simulation parameters:
n_warmup = 5000
box_size = 1.5
max_speed = 1
rng = np.random.default_rng(seed=0)

position, velocity, time = generate_random_walk(
    T=4000, dt=dt, boxsize=box_size, rng=rng, ndim=ndim, max_speed = max_speed
)
## Warm up

s = rng.uniform(size=(n,)*ndim) * 0.01
for step_iter in range(n_warmup):
    s = step(s, np.zeros(ndim))


n_steps = position.shape[0]
population_records = 1000
recorded_population = np.zeros((population_records,*((n,)*ndim)))
recorded_sample_neuron = np.zeros(n_steps)

population_recording_index = 0
for step_iter in tqdm.tqdm(range(1, n_steps), "Running steps"):
    s = step(s, velocity[step_iter])
    if (
        step_iter % (n_steps // population_records) == 0
        and population_recording_index < recorded_population.shape[0]
    ):
        recorded_population[population_recording_index] = s.copy()

    recorded_sample_neuron[step_iter] = s[*((0,)*ndim)]


np.save("simulation_data/3d_recorded_sample_neuron.npy", recorded_sample_neuron)
np.save("simulation_data/3d_recorded_position.npy", position)
np.save("simulation_data/3d_recorded_time.npy", time)
np.save("simulation_data/3d_recorded_population.npy", recorded_population)
