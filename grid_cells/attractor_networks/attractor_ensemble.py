import numpy as np
from .bump_grids import HeadDirection
from typing import Dict, Tuple
from tqdm import tqdm


class BatHeadDirectionSystem:
    def __init__(
        self,
        n_conjunctive=5,
        n_azimuth=256,
        n_pitch=256,
        tau=10e-3,
        dt=0.5e-3,
        intrinsic_noise=0,
        input_noise=0,
        size=1,
        rng: np.random.Generator = None,
        **kwargs,
    ):
        rng = rng or np.random.default_rng(seed=0)
        self.azimuth_ring = HeadDirection(
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
        self.pitch_ring = HeadDirection(
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
        self.conjunctive_neurons = np.zeros(n_conjunctive, dtype=float)
        self.weight_matrix = np.zeros((n_conjunctive, n_azimuth, n_pitch))
        connected_neurons = rng.integers(
            0, (n_azimuth, n_pitch), size=(n_conjunctive, 2)
        )
        for neuron_index, synapse_pair in enumerate(connected_neurons):
            self.weight_matrix[neuron_index, synapse_pair[0], synapse_pair[1]] = 1
        self.intrinsic_noise = intrinsic_noise
        self.input_noise = input_noise

    def warm_up(self, tol=1e-5, max_iter=100000, azimuth_pos=None, pitch_pos=None):
        """Relax the network until consecutive states differ by less than ``tol``."""
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

    def step(self, v_azimuth=0, v_pitch=0, pos_azimuth=None, pos_pitch=None):
        intrinsic_noise_term = 0
        if self.intrinsic_noise:
            noise_amplitude = self.intrinsic_noise * np.sqrt(self.dt / self.tau)
            intrinsic_noise_term = noise_amplitude * self.rng.normal(size=self.s.shape)

        total_input = (
            np.dot(np.dot(self.weight_matrix, self.pitch_ring.s), self.azimuth_ring.s)
            + 1
        )
        rate_derivatives = -self.conjunctive_neurons + np.maximum(total_input, 0.0)

        self.conjunctive_neurons = (
            self.conjunctive_neurons
            + (self.dt / self.tau) * rate_derivatives
            + intrinsic_noise_term
        )
        self.azimuth_ring.step(v_azimuth, v_pitch)
        self.pitch_ring.step(v_pitch, pos_pitch)

    def run_simulation(
        self,
        v: np.ndarray,
    ) -> Dict[str, np.ndarray]:

        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError("Velocity input must match dimension of attractor network")

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

        for step_iter in tqdm(range(n_steps), desc="Running Simulation Steps"):
            (self.step(*v[step_iter]))
            output_dict["conj_cells"][step_iter] = self.conjunctive_neurons
            output_dict["azimuth_cells"][step_iter] = np.array(
                [
                    self.azimuth_ring.s[tuple(cell_index)]
                    for cell_index in azimuth_rec_cells
                ]
            )
            output_dict["pitch_cells"][step_iter] = np.array(
                [
                    self.azimuth_ring.s[tuple(cell_index)]
                    for cell_index in pitch_rec_cells
                ]
            )
        return output_dict
