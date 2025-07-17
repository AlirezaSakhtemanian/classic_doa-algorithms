import numpy as np
from scipy import linalg
from signal_model.sensor_array import UniformLinearSensorArray


class Music(UniformLinearSensorArray):
    def __init__(
        self,
        all_doas: list,
        num_subarray: int = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.all_doas = all_doas
        self.num_subarray = num_subarray
        self.steering_matrix_cache = None

    def _manifold_matrix(self, num_antenna: int = None) -> np.ndarray:
        if num_antenna is None:
            num_antenna = self.num_antenna
        if self.steering_matrix_cache is None or self.steering_matrix_cache_key != num_antenna:
            self.steering_matrix_cache = self.steering_matrix(self.all_doas, num_antenna)
            self.steering_matrix_cache_key = num_antenna
        return self.steering_matrix_cache

    def estimate(self, input_signal: np.ndarray, num_sources: int) -> np.ndarray:
        R = np.cov(input_signal, rowvar=False)
        noise_subspace = linalg.svd(R)[0]
        noise_subspace = noise_subspace[:, num_sources:]
        noise_subspace = noise_subspace @ noise_subspace.conj().T

        if self.num_subarray is not None:
            num_antenna_adj = self.num_antenna - (self.num_subarray - 1)
        else:
            num_antenna_adj = self.num_antenna

        A = self._manifold_matrix(num_antenna_adj)
        p_music = A.conj().T @ noise_subspace @ A
        p_music = np.diag(p_music)
        p_music = np.where(p_music <= 0, 1e-6, p_music)
        return np.abs(1 / p_music)

    def estimate_via_noise_subspace(self, noise_subspace: np.ndarray) -> np.ndarray:
        noise_subspace = noise_subspace @ noise_subspace.conj().T

        if self.num_subarray is not None:
            num_antenna -= (self.num_subarray - 1)
        else:
            num_antenna = self.num_antenna

        A = self._manifold_matrix(num_antenna=num_antenna)
        p_music = A.conj().T @ noise_subspace @ A
        p_music = np.diag(p_music)
        p_music = np.where(p_music <= 0, 1e-6, p_music)
        return np.abs(1 / p_music)
