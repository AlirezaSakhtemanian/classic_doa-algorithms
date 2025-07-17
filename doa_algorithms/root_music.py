import numpy as np
from scipy import linalg
from signal_model.antenna_response import FarField1DSource


class RootMUSIC(FarField1DSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def estimate(self, input_signal: np.ndarray):
        if input_signal.shape[1] != self.num_antenna:
            raise ValueError(
                f"Input signal should have {self.num_antenna} columns (antennas). "
                f"Got shape {input_signal.shape}"
            )

        R = np.cov(input_signal.T)
        eigenvectors = linalg.svd(R)[0]

        if self.num_target >= self.num_antenna:
            raise ValueError(
                f"Number of sources ({self.num_target}) must be less than "
                f"number of antennas ({self.num_antenna})"
            )

        noise_eigenvectors = eigenvectors[:, self.num_target:]
        Q = noise_eigenvectors @ noise_eigenvectors.conj().T
        a_coeffs = np.zeros(2 * self.num_antenna - 1, dtype=complex)
        for i in range(self.num_antenna):
            for j in range(self.num_antenna):
                a_coeffs[self.num_antenna - 1 + i - j] += Q[i, j]

        all_roots = np.roots(a_coeffs)
        sorted_indices = np.argsort(np.abs(np.abs(all_roots) - 1.0))
        signal_roots = all_roots[sorted_indices[:self.num_target]]
        angles = np.angle(signal_roots)
        sin_thetas = -angles * self._lambda / (2 * np.pi * self.d)
        valid_indices = np.abs(sin_thetas) <= 1.0
        sin_thetas = sin_thetas[valid_indices]

        if len(sin_thetas) == 0:
            return np.array([])

        return np.arcsin(sin_thetas)
