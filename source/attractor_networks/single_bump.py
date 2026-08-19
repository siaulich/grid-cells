import numpy as np
import sympy as sy


class ToroidBurakFiete2009:
    def __init__(
        self,
        n=64,
        tau=10e-3,
        dt=0.5e-3,
        periodicity=13,
        rng=np.random.default_rng(seed=69),
    ):
        self.n = n
        self.tau = tau
        self.dt = dt

        #self.beta = 3.0 / periodicity**2
        self.gamma = 0.1 / n
        self.a_weight = 1
        self.inhibition = 1

        self.l_shift = 2.0
        self.alpha = 0.10315

        self.B0 = 1
        self.rng = rng
        self.dir_vectors = {
            "E": np.array([1.0, 0.0]),
            "W": np.array([-1.0, 0.0]),
            "N": np.array([0.0, 1.0]),
            "S": np.array([0.0, -1.0]),
        }
        dir_pattern = np.array([["E", "N"], ["S", "W"]])
        theta_dir = np.tile(dir_pattern, (n // 2, n // 2))

        self.directed_kernels = {}
        self.directed_masks = {}

        for preferred_direction in self.dir_vectors:
            self.directed_kernels[preferred_direction] = np.fft.fft2(
                self.periodic_kernel(
                    n,
                    e_theta=self.dir_vectors[preferred_direction],
                )
            )
            self.directed_masks[preferred_direction] = theta_dir == preferred_direction
        self.e_theta_x = np.vectorize(lambda d: self.dir_vectors[d][0])(theta_dir)
        self.e_theta_y = np.vectorize(lambda d: self.dir_vectors[d][1])(theta_dir)

        self.s = rng.uniform(size=(n, n)) * 0.01
        self.s[int(n // 2), int(n // 2)] = 0.5

    def periodic_kernel(self, n,  e_theta=(0.0, 0.0)):
        idx = np.arange(n)
        d = idx - n // 2
        d = np.where(d > n / 2, d - n, d)
        d = np.where(d < -n / 2, d + n, d)
        dx, dy = np.meshgrid(d, d, indexing="ij")

        sx = dx - self.l_shift * e_theta[0]
        sy = dy - self.l_shift * e_theta[1]
        r2 = sx**2 + sy**2

        K = self.a_weight * np.exp(-self.gamma * r2) - self.inhibition
        K = np.fft.ifftshift(K)
        return K

    def recurrent_input(self, s):
        rec_input = np.zeros_like(s)
        for dir in self.dir_vectors:
            rec_input += np.real(
                np.fft.ifft2(
                    np.fft.fft2(self.directed_masks[dir] * s)
                    * self.directed_kernels[dir]
                )
            )
        return rec_input

    def feedforward_input(self, vx, vy):
        return self.B0 * (
            1.0 + self.alpha * (self.e_theta_x * vx + self.e_theta_y * vy)
        )

    def step(self, vx=0, vy=0):
        total_input = self.recurrent_input(self.s) + self.feedforward_input(vx, vy)
        self.s = self.s + (self.dt / self.tau) * (
            -self.s + np.maximum(total_input, 0.0)
        )

    def warm_up(self, tol=1e-5, max_iter=100000):
        prev_net_state = self.s.copy()
        self.step()
        step = 0
        while np.max(np.abs(prev_net_state - self.s)) > tol:
            prev_net_state = self.s.copy()
            self.step()
            if step >= max_iter:
                raise RuntimeError("Exceed the prescribed recursion depth")
            step += 1
        print(f"Ran warm up for {step} steps")


class ToroidZhang1996:
    def __init__(
        self,
        n=64,
        tau=10e-3,
        dt=0.5e-3,
        revolutions=1,
        rng=np.random.default_rng(seed=69),
    ):
        self.n = n
        self.tau = tau
        self.dt = dt

        self.beta = 0.1 / n
        self.inhibition = 1
        self.a_weight = 1

        self.l_shift = self._compute_l_shift(revolutions)
        self.B0 = 1.0

        K_sym, K_asym_x, K_asym_y = self._build_kernels(n)
        self.K_sym_fft = np.fft.fft2(K_sym)
        self.K_asym_x_fft = np.fft.fft2(K_asym_x)
        self.K_asym_y_fft = np.fft.fft2(K_asym_y)

        self.s = rng.uniform(size=(n, n)) * 0.001
        self.s[n // 2, n // 2] = 0.5

    def _distance_grid(self, n):
        idx = np.arange(n)
        d = idx - n // 2
        d = np.where(d > n / 2, d - n, d)
        d = np.where(d < -n / 2, d + n, d)
        dx, dy = np.meshgrid(d, d, indexing="ij")
        return dx, dy

    def _compute_l_shift(self, revolutions=1):
        target_gain = self.n / (2 * np.pi) * revolutions
        idx = np.arange(self.n)
        d = idx - self.n // 2
        d = np.where(d > self.n / 2, d - self.n, d)
        d = np.where(d < -self.n / 2, d + self.n, d)
        dxg, dyg = np.meshgrid(d, d, indexing="ij")
        r2 = dxg**2 + dyg**2

        K_sym = self.a_weight * np.exp(-self.beta * r2) - self.inhibition
        common = self.a_weight * self.beta * np.exp(-self.beta * r2)
        dK_dx = -2 * dxg * common

        norm = np.max(np.abs(K_sym))
        dnorm = np.max(np.abs(dK_dx))
        return target_gain * self.tau * dnorm / norm

    def _build_kernels(self, n):
        dx, dy = self._distance_grid(n)
        r2 = dx**2 + dy**2

        K_sym = self.a_weight * np.exp(-self.beta * r2) - self.inhibition

        common = -self.a_weight * self.beta * np.exp(-self.beta * r2)
        dK_dx = -2 * dx * common
        dK_dy = -2 * dy * common

        norm = np.max(np.abs(K_sym))
        dnorm = max(np.max(np.abs(dK_dx)), np.max(np.abs(dK_dy)), 1e-12)
        K_asym_x = self.l_shift * (dK_dx / dnorm) * norm
        K_asym_y = self.l_shift * (dK_dy / dnorm) * norm

        K_sym = np.fft.ifftshift(K_sym)
        K_asym_x = np.fft.ifftshift(K_asym_x)
        K_asym_y = np.fft.ifftshift(K_asym_y)
        return K_sym, K_asym_x, K_asym_y

    def recurrent_input(self, s, vx, vy):
        s_fft = np.fft.fft2(s)
        rec = np.real(np.fft.ifft2(s_fft * self.K_sym_fft))
        if vx != 0:
            rec += vx * np.real(np.fft.ifft2(s_fft * self.K_asym_x_fft))
        if vy != 0:
            rec += vy * np.real(np.fft.ifft2(s_fft * self.K_asym_y_fft))
        return rec

    def feedforward_input(self):
        return self.B0

    def step(self, vx=0, vy=0):
        total_input = self.recurrent_input(self.s, vx, vy) + self.feedforward_input()
        self.s = self.s + (self.dt / self.tau) * (
            -self.s + np.maximum(total_input, 0.0)
        )

    def warm_up(self, tol=1e-6, max_iter=100000):
        prev_net_state = self.s.copy()
        self.step()
        step = 0
        while np.max(np.abs(prev_net_state - self.s)) > tol:
            prev_net_state = self.s.copy()
            self.step()
            if step >= max_iter:
                raise RuntimeError("Exceed the prescribed recursion depth")
            step += 1
        print(f"Ran warm up for {step} steps")
