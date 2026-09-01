"""Periodic bump-attractor network models.

The classes in this module simulate grid-cell activity on a two-dimensional
toroidal lattice.
"""

import numpy as np
from typing import Tuple, Callable, Dict
from tqdm import tqdm


class AttractorNetworkBase:
    def __init__(
        self,
        n=64,
        tau=10e-3,
        dt=0.5e-3,
        intrinsic_noise=0,
        input_noise=0,
        periodicity=20,
        use_single_bump=False,
        rng=None,
        **kwargs,
    ):
        self.n = n
        self.tau = tau
        self.dt = dt
        self.rng = rng or np.random.default_rng(42)
        self.intrinsic_noise = intrinsic_noise
        self.input_noise = input_noise

        if not use_single_bump:
            beta = 3.0 / periodicity**2
            gamma = 1.05 * beta
            a_weight = 1
            a_weight = 1
            self.kernel_func = lambda r2: a_weight * np.exp(-gamma * r2) - np.exp(
                -beta * r2
            )
            self.kernel_deriv_func = lambda r2: a_weight * gamma * np.exp(
                -gamma * r2
            ) - beta * np.exp(-beta * r2)

        else:
            gamma = 0.1 / n
            a_weight = 1
            inhibition = 1
            self.kernel_func = lambda r2: a_weight * np.exp(-gamma * r2) - inhibition
            self.kernel_deriv_func = lambda r2: a_weight * gamma * np.exp(-gamma * r2)

        self.s = self.rng.uniform(size=(self.n, self.n)) * 0.1
        self.anchor_points = []
        self._setup_attractor(**kwargs)

    def add_anchor_point(
        self,
        weight_func: Callable = lambda position: np.exp(
            np.sum(np.square(position - [1, 0, 0])) / 0.5
        ),
        mask=None,
        cell_index=(0, 0),
        strength=0.1,
    ):
        if len(cell_index) != 2:
            raise ValueError("cell_index has to be an (i,j) index pair")
        i, j = cell_index
        if not (0 <= i < self.n) or not (0 <= j < self.n):
            raise ValueError("indices must be smaller than network size per dimension")
        if mask is None:
            mask = np.zeros_like(self.s)
            mask[i, j] = 1

        self.anchor_points.append(lambda pos: strength * weight_func(pos) * mask)

    def step(self, vx=0, vy=0, pos=None, intrinsic_noise=None, input_noise=None):
        """Advance the activity state by one Euler integration step."""

        eff_input_noise = input_noise if input_noise is not None else self.input_noise
        eff_intrinsic_noise = (
            intrinsic_noise if intrinsic_noise is not None else self.intrinsic_noise
        )

        if eff_input_noise:
            vx += (
                eff_input_noise
                * np.sqrt(self.dt / self.tau)
                * self.rng.normal()
                * np.abs(vx)
            )
            vy += (
                eff_input_noise
                * np.sqrt(self.dt / self.tau)
                * self.rng.normal()
                * np.abs(vy)
            )

        total_input = self._recurrent_input(self.s, vx, vy) + self._feedforward_input(
            self.s, vx, vy
        )
        anchor_input = self._compute_anchor_input(pos) if pos is not None else 0.0

        rate_derivatives = -self.s + np.maximum(total_input, 0.0) + anchor_input

        intrinsic_noise_term = 0.0
        if eff_intrinsic_noise:
            noise_amplitude = eff_intrinsic_noise * np.sqrt(self.dt / self.tau)
            intrinsic_noise_term = noise_amplitude * self.rng.normal(size=self.s.shape)

        self.s = self.s + (self.dt / self.tau) * rate_derivatives + intrinsic_noise_term

    def _compute_anchor_input(self, pos):
        anchor_input = np.zeros_like(self.s)
        for anchor in self.anchor_points:
            anchor_input += anchor(pos)
        return anchor_input

    def warm_up(self, tol=1e-5, max_iter=100000, pos=None):
        """Relax the network until consecutive states differ by less than ``tol``."""
        prev_net_state = self.s.copy()
        self.step()
        step = 0
        while np.max(np.abs(prev_net_state - self.s)) > tol:
            prev_net_state = self.s.copy()
            self.step(pos=pos, intrinsic_noise=0, input_noise=0)
            if step >= max_iter:
                raise RuntimeError("Exceed the prescribed recursion depth")
            step += 1
        print(f"Ran warm up for {step} steps")

    def _add_variables(self, output_dict, n_steps):
        pass

    def _record_variables(self, output_dict, step_iter):
        pass

    def run_simulation(
        self, v: np.ndarray, pos=None, rec_cells=None, n_snapshots=1000
    ) -> Dict[str, np.ndarray]:
        """Run the network simulation with given velocity input.
        
        Parameters
        ----------
        v : np.ndarray
            Velocity input array of shape (n_steps, 2) with vx and vy components.
        pos : np.ndarray, optional
            Position array of shape (n_steps, 2). If provided, passed to step().
        rec_cells : list, optional
            List of cell indices to record. Defaults to 9 random cells if None.
        n_snapshots : int, optional
            Number of population snapshots to record (default: 1000).
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'cell_recording': recorded activity of selected cells
            - 'cell_indices': indices of recorded cells
            - 'popuplation_snapshots': snapshots of full network state
            - 'snapshot_indices': time indices of snapshots
        """
        rec_cells = rec_cells or list(np.random.randint(0, self.n - 1, size=(9, 2)))
        if np.any(np.array(rec_cells) >= self.n) or np.any(np.array(rec_cells) < 0):
            raise ValueError(f"Indices of recorded cells must be in {0}...{self.n}")
        if v.ndim != 2:
            raise ValueError("Input proper velocity input")
        n_steps = v.shape[0]
        n_cell_records = len(rec_cells)

        output_dict = {}
        output_dict["cell_recording"] = np.zeros((n_steps, n_cell_records))
        output_dict["cell_indices"] = rec_cells
        output_dict["popuplation_snapshots"] = np.zeros((n_snapshots, self.n, self.n))
        output_dict["snapshot_indices"] = np.linspace(
            0, n_steps - 1, n_snapshots, dtype=int
        )
        self._add_variables(output_dict, n_steps)
        snapshot_iter = 0

        for step_iter in tqdm(range(n_steps), desc="Running Simulation Steps"):
            (
                self.step(*v[step_iter], pos=pos[step_iter])
                if pos is not None
                else self.step(*v[step_iter])
            )
            output_dict["cell_recording"][step_iter] = np.array(
                [self.s[*cell_index] for cell_index in output_dict["cell_indices"]]
            )
            if step_iter == output_dict["snapshot_indices"][snapshot_iter]:
                output_dict["popuplation_snapshots"][snapshot_iter] = self.s
                snapshot_iter += 1
            self._record_variables(output_dict, step_iter)
        return output_dict

    def _recurrent_input(self, s, vx, vy):
        """Compute recurrent input from network state and velocity.
        
        Parameters
        ----------
        s : np.ndarray
            Network state (activity).
        vx : float
            X-component of velocity.
        vy : float
            Y-component of velocity.

        Returns
        -------
        np.ndarray
            Recurrent input to the network.
        """
        raise NotImplementedError("Has to be implemented by subclasses")

    def _feedforward_input(self, s, vx, vy):
        """Compute feedforward input from velocity.
        
        Parameters
        ----------
        s : np.ndarray
            Network state (activity).
        vx : float
            X-component of velocity.
        vy : float
            Y-component of velocity.
        
        Returns
        -------
        np.ndarray
            Feedforward input to the network.
        """
        raise NotImplementedError("Has to be implemented by subclasses")

    def _setup_attractor(self, **kwargs):
        """Set up attractor-specific parameters.
        
        This method should be overridden by subclasses to initialize
        any additional attractor-specific parameters or kernels.
        """
        print("WARNING: No additional setup for attractor was implemented")
        pass


class ToroidBurakFiete2009(AttractorNetworkBase):
    """Burak--Fiete (2009) attractor network with directional kernels.

    Parameters control the lattice size, integration timestep, kernel
    periodicity, and whether a single bump should be supported.  The public
    state is stored in :attr:`s` and can be advanced with :meth:`step`.
    To establish the grid pattern, a warm up with :meth:`warmup` is recommended.
    """

    def _setup_attractor(self, **kwargs):
        self.l_shift = 2.0
        self.alpha = 0.10315

        self.B0 = 1
        self.dir_vectors = {
            "E": np.array([1.0, 0.0]),
            "W": np.array([-1.0, 0.0]),
            "N": np.array([0.0, 1.0]),
            "S": np.array([0.0, -1.0]),
        }
        dir_pattern = np.array([["E", "N"], ["S", "W"]])
        theta_dir = np.tile(dir_pattern, (self.n // 2, self.n // 2))

        self.directed_kernels = {}
        self.directed_masks = {}

        for preferred_direction in self.dir_vectors:
            self.directed_kernels[preferred_direction] = np.fft.fft2(
                self._periodic_kernel(
                    self.n,
                    e_theta=self.dir_vectors[preferred_direction],
                )
            )
            self.directed_masks[preferred_direction] = theta_dir == preferred_direction
        self.e_theta_x = np.vectorize(lambda d: self.dir_vectors[d][0])(theta_dir)
        self.e_theta_y = np.vectorize(lambda d: self.dir_vectors[d][1])(theta_dir)

    def _periodic_kernel(self, n, e_theta=(0.0, 0.0)):
        """Return a periodic kernel shifted in direction ``e_theta``."""
        idx = np.arange(n)
        d = idx - n // 2
        d = np.where(d > n / 2, d - n, d)
        d = np.where(d < -n / 2, d + n, d)
        dx, dy = np.meshgrid(d, d, indexing="ij")

        sx = dx - self.l_shift * e_theta[0]
        sy = dy - self.l_shift * e_theta[1]
        r2 = sx**2 + sy**2

        K = self.kernel_func(r2)
        K = np.fft.ifftshift(K)
        return K

    def _recurrent_input(self, s, vx, vy):
        """Calculate recurrent input for activity array ``s``."""
        rec_input = np.zeros_like(s)
        for dir in self.dir_vectors:
            rec_input += np.real(
                np.fft.ifft2(
                    np.fft.fft2(self.directed_masks[dir] * s)
                    * self.directed_kernels[dir]
                )
            )
        return rec_input

    def _feedforward_input(self, s, vx, vy):
        """Calculate velocity-dependent input for velocity ``(vx, vy)``."""
        return self.B0 * (
            1.0 + self.alpha * (self.e_theta_x * vx + self.e_theta_y * vy)
        )


class ToroidZhang1996(AttractorNetworkBase):
    """Zhang (1996) attractor network with symmetric/asymmetric kernels.

    Velocity input is coupled to spatial derivatives of the recurrent kernel;
    the resulting activity state is available in :attr:`s` and can be advanced with :meth:`step`.
    To establish the grid pattern, a warm up with :meth:`warmup` is recommended.
    """

    def _setup_attractor(self, revolutions=1):
        self.speed_modulation = self._compute_speed_modulation(revolutions)
        self.B0 = 1.0

        K_sym, K_asym_x, K_asym_y = self._build_kernels(self.n)
        self.K_sym_fft = np.fft.fft2(K_sym)
        self.K_asym_x_fft = np.fft.fft2(K_asym_x)
        self.K_asym_y_fft = np.fft.fft2(K_asym_y)

    def _distance_grid(self, n):
        """Build wrapped x and y displacement grids for an ``n``-cell lattice."""
        idx = np.arange(n)
        d = idx - n // 2
        d = np.where(d > n / 2, d - n, d)
        d = np.where(d < -n / 2, d + n, d)
        dx, dy = np.meshgrid(d, d, indexing="ij")
        return dx, dy

    def _compute_speed_modulation(self, revolutions=1):
        """Compute the asymmetric-kernel scale for ``revolutions`` per cycle."""
        target_gain = self.n / (2 * np.pi) * revolutions
        idx = np.arange(self.n)
        d = idx - self.n // 2
        d = np.where(d > self.n / 2, d - self.n, d)
        d = np.where(d < -self.n / 2, d + self.n, d)
        dxg, dyg = np.meshgrid(d, d, indexing="ij")
        r2 = dxg**2 + dyg**2

        K_sym = self.kernel_func(r2)
        common = self.kernel_deriv_func(r2)
        dK_dx = -2 * dxg * common  # / np.sqrt(r2 + 1e-6)

        norm = np.max(np.abs(K_sym))
        dnorm = np.max(np.abs(dK_dx))
        return target_gain * self.tau * dnorm / norm

    def _build_kernels(self, n):
        """Construct centered symmetric and velocity-dependent kernels."""
        dx, dy = self._distance_grid(n)
        r2 = dx**2 + dy**2

        K_sym = self.kernel_func(r2)

        common = self.kernel_deriv_func(r2)

        dK_dx = -2 * dx * common  # / np.sqrt(r2 + 1e-6)
        dK_dy = -2 * dy * common  # / np.sqrt(r2 + 1e-6)

        norm = np.max(np.abs(K_sym))
        dnorm = max(np.max(np.abs(dK_dx)), np.max(np.abs(dK_dy)), 1e-12)
        K_asym_x = self.speed_modulation * dK_dx * (norm / dnorm)
        K_asym_y = self.speed_modulation * dK_dy * (norm / dnorm)

        K_sym = np.fft.ifftshift(K_sym)
        K_asym_x = np.fft.ifftshift(K_asym_x)
        K_asym_y = np.fft.ifftshift(K_asym_y)
        return K_sym, K_asym_x, K_asym_y

    def _recurrent_input(self, s, vx, vy):
        """Calculate recurrent input for state ``s`` and velocity ``(vx, vy)``."""
        s_fft = np.fft.fft2(s)
        rec = np.real(np.fft.ifft2(s_fft * self.K_sym_fft))
        if vx != 0:
            rec += vx * np.real(np.fft.ifft2(s_fft * self.K_asym_x_fft))
        if vy != 0:
            rec += vy * np.real(np.fft.ifft2(s_fft * self.K_asym_y_fft))
        return rec

    def _feedforward_input(self, s, vx, vy):
        return self.B0


class HeadDirection(ToroidZhang1996):

    def _add_variables(self, output_dict, n_steps):
        output_dict["decoded_angle"] = np.zeros((n_steps, 2))
        output_dict["anchor_input"] = np.zeros(n_steps)

    def _record_variables(self, output_dict, step_iter):
        output_dict["decoded_angle"][step_iter] = self.decode_orientation()

    def decode_orientation(self):
        """
        Decodes the estimated orientation angles from the 2D network sheet.

        Returns:
        - np.array([angle_x, angle_y]): Estimated angles in radians [0, 2*pi).
        """
        s_2d = self.s
        nx, ny = s_2d.shape

        x_phases = 2 * np.pi * np.arange(nx) / nx
        y_phases = 2 * np.pi * np.arange(ny) / ny

        profile_x = np.sum(s_2d, axis=1)
        profile_y = np.sum(s_2d, axis=0)

        mean_x_angle = np.angle(np.sum(profile_x * np.exp(1j * x_phases)))
        mean_y_angle = np.angle(np.sum(profile_y * np.exp(1j * y_phases)))

        angle_1 = (-mean_x_angle + 2 * np.pi) % (2 * np.pi)
        angle_2 = (-mean_y_angle + 2 * np.pi) % (2 * np.pi)

        return np.array([angle_1, angle_2])

    def encode_orientation(self, target_angles, width=0.3):
        """
        Generates the corresponding neural activity state (a 2D bump)
        for a given set of target orientation angles.

        Parameters:
        - target_angles: np.array or list of [angle_x, angle_y] in radians.
        - width: Controls the spatial width (spread) of the activity bump.

        Returns:
        - s_2d: np.array of shape (nx, ny) representing the network activity state.
        """
        nx, ny = self.s.shape
        target_x, target_y = target_angles

        x_phases = 2 * np.pi * np.arange(nx) / nx
        y_phases = 2 * np.pi * np.arange(ny) / ny
        X, Y = np.meshgrid(x_phases, y_phases, indexing="ij")

        dx = np.arctan2(np.sin(-(X + target_x)), np.cos(-(X + target_x)))
        dy = np.arctan2(np.sin(-(Y + target_y)), np.cos(-(Y + target_y)))
        s_2d = np.exp(-(dx**2 + dy**2) / (2 * width**2))

        return s_2d
