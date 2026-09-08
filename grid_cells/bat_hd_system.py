import numpy as np
import numpy as np
from typing import Union, Tuple, Dict
from tqdm import tqdm
from .plot_tools import angular_error


class HeadDirectionNetwork:
    def __init__(
        self,
        n=64,
        tau=10e-3,
        dt=0.5e-3,
        intrinsic_noise=0,
        input_noise=0,
        size=1,
        use_single_bump=False,
        rng: np.random.Generator = None,
        **kwargs,
    ):

        self.shape = (int(n),) if np.isscalar(n) else tuple(map(int, n))
        if not self.shape or any(size < 1 for size in self.shape):
            raise ValueError("n must be a positive integer or a sequence of integers")
        self.ndim = len(self.shape)
        self.tau = tau
        self.dt = dt
        self.rng = rng or np.random.default_rng(42)
        self.intrinsic_noise = intrinsic_noise
        self.input_noise = input_noise

        if isinstance(size, (float, int)):
            size = np.ones((self.ndim,), dtype=float) * size
        elif isinstance(size, (tuple, list)):
            size = np.array(size)
        else:
            raise ValueError("WTF?")

        if not use_single_bump:
            beta = 0.01 * size
            gamma = 1.05 * beta
            a_weight = 1
            a_weight = 1
            self.kernel_func = lambda dx: a_weight * np.exp(
                -np.dot(dx, gamma)
            ) - np.exp(-np.dot(dx, beta))
            self.kernel_deriv_func = (
                lambda dx: a_weight
                * gamma.reshape((1,) * self.ndim + (-1,))
                * np.exp(-np.dot(dx, gamma)[..., np.newaxis])
                - beta.reshape((1,) * self.ndim + (-1,))
                * np.exp(-np.dot(dx, beta))[..., np.newaxis]
            )

        else:
            gamma = 0.01 * size / n
            a_weight = 1
            inhibition = 1
            self.kernel_func = (
                lambda dx: a_weight * np.exp(-np.dot(dx, gamma)) - inhibition
            )
            self.kernel_deriv_func = (
                lambda dx: a_weight
                * gamma.reshape((1,) * self.ndim + (-1,))
                * np.exp(-np.dot(dx, gamma))[..., np.newaxis]
            )

        self.s = self.rng.uniform(size=self.shape) * 0.1
        self._setup_attractor(**kwargs)

    def step(
        self, v=0, pos=None, intrinsic_noise=None, input_noise=None, anchor_input=None
    ):
        """Advance the activity state by one Euler integration step.

        Parameters
        ----------
        v : scalar or array-like
            Velocity vector. A scalar is broadcast to every dimension.
        pos : array-like, optional
            Position passed to registered anchor points.
        intrinsic_noise, input_noise : float, optional
            Per-step noise scales overriding the values configured at init.
        """

        eff_input_noise = input_noise if input_noise is not None else self.input_noise
        eff_intrinsic_noise = (
            intrinsic_noise if intrinsic_noise is not None else self.intrinsic_noise
        )

        if np.isscalar(v):
            v = np.ones((self.ndim)) * v
        else:
            v = np.asarray(v)

        if eff_input_noise:
            v += (
                eff_input_noise
                * np.sqrt(self.dt / self.tau)
                * self.rng.normal(size=v.shape)
                * np.abs(v)
            )

        total_input = self._recurrent_input(self.s, v) + self._feedforward_input(
            self.s, v
        )
        if anchor_input is not None:
            total_input += anchor_input

        rate_derivatives = -self.s + np.maximum(total_input, 0.0)

        intrinsic_noise_term = 0.0
        if eff_intrinsic_noise:
            noise_amplitude = eff_intrinsic_noise * np.sqrt(self.dt / self.tau)
            intrinsic_noise_term = noise_amplitude * self.rng.normal(size=self.s.shape)

        self.s = self.s + (self.dt / self.tau) * rate_derivatives + intrinsic_noise_term

    def warm_up(self, tol=1e-5, max_iter=100000, pos=None):
        """Relax the network until consecutive states differ by less than ``tol``."""
        prev_net_state = self.s.copy()
        self.step(pos=pos, intrinsic_noise=0, input_noise=0)
        step = 0
        while np.max(np.abs(prev_net_state - self.s)) > tol:
            prev_net_state = self.s.copy()
            self.step(pos=pos, intrinsic_noise=0, input_noise=0)
            if step >= max_iter:
                raise RuntimeError("Exceed the prescribed recursion depth")
            step += 1
        print(f"Ran warm up for {step} steps")

    def _setup_attractor(self, revolutions: Union[Tuple, float] = 1):
        self.speed_modulation = self._compute_speed_modulation(revolutions)
        self.B0 = 1.0

        K_sym, K_asym = self._build_kernels(self.shape)
        self.K_sym_fft = np.fft.fftn(K_sym)
        self.K_asym_fft = np.fft.fftn(K_asym, axes=tuple(np.arange(K_asym.ndim - 1)))

    def _distance_grid(self, shape):
        """Build wrapped x and y displacement grids for an ``n``-cell lattice."""
        dist_list = []
        for n in shape:
            idx = np.arange(n)
            d = idx - n // 2
            d = np.where(d > n / 2, d - n, d)
            d = np.where(d < -n / 2, d + n, d)
            dist_list.append(d)
        dx = np.meshgrid(*dist_list, indexing="ij")
        dx = np.stack(dx, axis=-1)
        return dx

    def _compute_speed_modulation(self, revolutions=1):
        """Compute the asymmetric-kernel scale for ``revolutions`` per cycle."""
        if isinstance(revolutions, (float, int)):
            revolutions = np.ones((self.ndim,), dtype=float) * revolutions
        elif isinstance(revolutions, tuple):
            revolutions = np.array(revolutions)
        else:
            raise ValueError("WTF?")

        target_gain = np.asarray(self.shape) / (2 * np.pi) * revolutions
        dist_list = []
        for n in self.shape:
            idx = np.arange(n)
            d = idx - n // 2
            d = np.where(d > n / 2, d - n, d)
            d = np.where(d < -n / 2, d + n, d)
            dist_list.append(d)

        dx = np.meshgrid(*dist_list, indexing="ij")
        dx = np.stack(dx, axis=-1)

        K_sym = self.kernel_func(dx**2)
        common = self.kernel_deriv_func(dx**2)
        dK_dx = -2 * dx * common

        norm = np.max(np.abs(K_sym))
        dnorm = np.max(np.abs(dK_dx), axis=tuple(np.arange(self.ndim)))
        return target_gain * self.tau * dnorm / norm

    def _build_kernels(self, shape):
        """Construct centered symmetric and velocity-dependent kernels."""
        dx = self._distance_grid(shape)

        K_sym = self.kernel_func(dx**2)
        common = self.kernel_deriv_func(dx**2)

        dK_dx = -2 * dx * common

        norm = np.max(np.abs(K_sym))
        dnorm = np.max(np.abs(dK_dx), axis=tuple(np.arange(self.ndim)))
        K_asym = self.speed_modulation * dK_dx * (norm / dnorm)

        K_sym = np.fft.ifftshift(K_sym)
        K_asym = np.fft.ifftshift(K_asym, axes=tuple(np.arange(self.ndim)))
        return K_sym, K_asym

    def _recurrent_input(self, s, v: np.ndarray):
        """Calculate recurrent input for state ``s`` and velocity ``(vx, vy)``."""
        s_fft = np.fft.fftn(s)
        rec = np.real(np.fft.ifftn(s_fft * self.K_sym_fft))
        if np.any(v != 0):
            rec += np.real(np.fft.ifftn(s_fft * np.dot(self.K_asym_fft, v)))
        return rec

    def _feedforward_input(self, s, v):
        return self.B0

    def _add_variables(self, output_dict, n_steps):
        output_dict["decoded_angle"] = np.zeros((n_steps, 2))
        output_dict["anchor_input"] = np.zeros(n_steps)

    def _record_variables(self, output_dict, step_iter):
        output_dict["decoded_angle"][step_iter] = self.decode_orientation()

    def decode_orientation(self):
        """Decode the activity bump position into one angle per axis.

        Returns:
        - np.array([angle_x, angle_y]): Estimated angles in radians [0, 2*pi).
        """
        s_2d = self.s
        shape = s_2d.shape

        angle_list = []
        for axis_iter, nx in enumerate(shape):
            x_phases = 2 * np.pi * np.arange(nx) / nx

            profile_x = np.sum(
                s_2d, axis=tuple([i for i in range(len(shape)) if i != axis_iter])
            )

            mean_x_angle = np.angle(np.sum(profile_x * np.exp(1j * x_phases)))

            angle_1 = (-mean_x_angle + 2 * np.pi) % (2 * np.pi)
            angle_list.append(angle_1)

        return np.array(angle_list)

    def encode_orientation(self, target_angles, width=0.3):
        """Generate a periodic Gaussian bump for target orientation angles.

        Parameters:
        - target_angles: np.array or list of [angle_x, angle_y] in radians.
        - width: Controls the spatial width (spread) of the activity bump.

        Returns:
        - s_2d: np.array of shape (nx, ny) representing the network activity state.
        """
        shape = self.s.shape
        target_angles = np.asarray(target_angles)[*((np.newaxis,) * (len(shape) + 1))]

        phases_list = []
        for nx in shape:
            x_phases = 2 * np.pi * np.arange(nx) / nx
            phases_list.append(x_phases)
        X = np.meshgrid(*phases_list, indexing="ij")
        X = np.stack(X, axis=-1)

        dx = np.arctan2(np.sin(-(X + target_angles)), np.cos(-(X + target_angles)))

        s_2d = np.exp(-(np.sum(dx**2, axis=-1)) / (2 * width**2))

        return s_2d


class BatHeadDirectionSystem:
    def __init__(
        self,
        n_conjunctive=5,
        n_azimuth=256,
        n_pitch=256,
        tau=10e-3,
        dt=0.5e-3,
        intrinsic_noise=0,
        input_noise=0.1,
        size=1,
        rng: np.random.Generator = None,
        **kwargs,
    ):
        rng = rng or np.random.default_rng(seed=0)
        self.azimuth_ring = HeadDirectionNetwork(
            (n_azimuth,),
            tau,
            dt,
            intrinsic_noise,
            input_noise,
            size,
            use_single_bump=True,
            rng=rng,
            **kwargs,
        )
        self.pitch_ring = HeadDirectionNetwork(
            (n_pitch,),
            tau,
            dt,
            intrinsic_noise,
            input_noise,
            size,
            use_single_bump=True,
            rng=rng,
            **kwargs,
        )
        self.tau = tau
        self.dt = dt
        self.n_conjunctive = n_conjunctive
        self.n_azimuth = n_azimuth
        self.n_pitch = n_pitch
        self.rng = rng
        self.activation_sigma = 0.1
        self.connectivity_sigma = 0.1
        self.forward_strength = 0.1
        self.backward_strength = 0.1
        self.feedback_strength = 1
        self.inhibition = 1
        self.conjunctive_neurons = np.zeros(n_conjunctive, dtype=float)
        self.azimuth_weight_vectors = np.zeros((n_conjunctive, n_azimuth), dtype=float)
        self.pitch_weight_vectors = np.zeros((n_conjunctive, n_pitch), dtype=float)
        self.conjunctive_angels = rng.random(size=(n_conjunctive, 2)) * 2 * np.pi

        for neuron_index, [azimuth, pitch] in enumerate(self.conjunctive_angels):
            self.azimuth_weight_vectors[neuron_index] = (
                self.azimuth_ring.encode_orientation(azimuth, self.connectivity_sigma)
            )
            self.pitch_weight_vectors[neuron_index] = (
                self.pitch_ring.encode_orientation(pitch, self.connectivity_sigma)
            )

        self.intrinsic_noise = intrinsic_noise
        self.input_noise = input_noise

    def activation_weight_func(
        self,
        position,
        anchor,
    ):
        err = angular_error(position, anchor)
        d2 = np.sum(err**2, axis=-1)
        return np.exp(-d2 / (2 * self.activation_sigma**2))

    def warm_up(
        self, tol=1e-5, max_iter=100000, initial_pos: np.ndarray = np.array([0, 0])
    ):
        """Relax the network until consecutive states differ by less than ``tol``."""
        azimuth_pos, pitch_pos = initial_pos[[0, 0]]
        self.azimuth_ring.s = self.azimuth_ring.encode_orientation(azimuth_pos)
        self.pitch_ring.s = self.pitch_ring.encode_orientation(pitch_pos)

        prev_azimuth_state = self.azimuth_ring.s.copy()
        self.azimuth_ring.step(pos=azimuth_pos, intrinsic_noise=0, input_noise=0)

        prev_pitch_state = self.pitch_ring.s.copy()
        self.pitch_ring.step(pos=pitch_pos, intrinsic_noise=0, input_noise=0)

        step = 0
        while (
            max(
                np.max(np.abs(prev_azimuth_state - self.azimuth_ring.s)),
                np.max(np.abs(prev_pitch_state - self.pitch_ring.s)),
            )
            > tol
        ):
            prev_azimuth_state = self.azimuth_ring.s.copy()
            prev_pitch_state = self.pitch_ring.s.copy()
            self.azimuth_ring.step(pos=azimuth_pos, intrinsic_noise=0, input_noise=0)
            self.pitch_ring.step(pos=pitch_pos, intrinsic_noise=0, input_noise=0)
            if step >= max_iter:
                raise RuntimeError("Exceed the prescribed recursion depth")
            step += 1
        print(f"Ran warm up for {step} steps")

    def step(
        self,
        v_azimuth: float = 0,
        v_pitch: float = 0,
        pos_azimuth: float = None,
        pos_pitch: float = None,
    ):
        intrinsic_noise_term = 0
        previous_conj_state = self.conjunctive_neurons
        if self.intrinsic_noise:
            noise_amplitude = self.intrinsic_noise * np.sqrt(self.dt / self.tau)
            intrinsic_noise_term = noise_amplitude * self.rng.normal(
                size=self.conjunctive_neurons.shape
            )
        if pos_azimuth is not None and pos_pitch is not None:
            feedback_vector = self.activation_weight_func(
                self.conjunctive_angels,
                np.array([pos_azimuth, pos_pitch])[np.newaxis, :],
            )
        else:
            feedback_vector = np.zeros_like(self.conjunctive_neurons)

        total_input = (
            (
                np.dot(self.pitch_weight_vectors, self.pitch_ring.s)
                + np.dot(self.azimuth_weight_vectors, self.azimuth_ring.s)
            )
            * self.forward_strength
            + feedback_vector * self.feedback_strength
            - self.inhibition
        )
        rate_derivatives = -self.conjunctive_neurons + np.maximum(total_input, 0.0)

        self.conjunctive_neurons = (
            self.conjunctive_neurons
            + (self.dt / self.tau) * rate_derivatives
            + intrinsic_noise_term
        )
        self.azimuth_ring.step(
            v_azimuth,
            anchor_input=self.backward_strength
            * np.dot(self.azimuth_weight_vectors.T, previous_conj_state),
        )
        self.pitch_ring.step(
            v_pitch,
            anchor_input=self.backward_strength
            * np.dot(self.pitch_weight_vectors.T, previous_conj_state),
        )

    def run_simulation(
        self,
        v: np.ndarray,
        dir: np.ndarray = None,
    ) -> Dict[str, np.ndarray]:

        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError("Velocity input must match dimension of attractor network")

        if dir is not None:
            if dir.ndim != 2 or dir.shape[1] != 2:
                raise ValueError(
                    "Velocity input must match dimension of attractor network"
                )

        n_steps = v.shape[0]

        output_dict = {}
        output_dict["conj_cells"] = np.zeros((n_steps, self.n_conjunctive), dtype=float)
        azimuth_rec_cells = list(self.rng.integers(0, self.n_azimuth, size=(9, 1)))
        pitch_rec_cells = list(self.rng.integers(0, self.n_pitch, size=(9, 1)))
        output_dict["azimuth_cells"] = np.zeros(
            (n_steps, len(azimuth_rec_cells)), dtype=float
        )
        output_dict["pitch_cells"] = np.zeros(
            (n_steps, len(pitch_rec_cells)), dtype=float
        )
        output_dict["decoded_angle"] = np.zeros((n_steps, 2), dtype=float)
        for step_iter in tqdm(range(n_steps), desc="Running Simulation Steps"):
            if dir is not None:
                self.step(*v[step_iter], *dir[step_iter])
            else:
                self.step(*v[step_iter])
            output_dict["conj_cells"][step_iter] = self.conjunctive_neurons
            output_dict["azimuth_cells"][step_iter] = np.array(
                [
                    self.azimuth_ring.s[tuple(cell_index)]
                    for cell_index in azimuth_rec_cells
                ]
            )
            output_dict["pitch_cells"][step_iter] = np.array(
                [self.pitch_ring.s[tuple(cell_index)] for cell_index in pitch_rec_cells]
            )
            output_dict["decoded_angle"][step_iter] = np.stack(
                [
                    self.azimuth_ring.decode_orientation(),
                    self.pitch_ring.decode_orientation(),
                ]
            ).flatten()

        return output_dict
