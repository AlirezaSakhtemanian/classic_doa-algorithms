import numpy as np
from signal_model.sensor_array import UniformLinearSensorArray


class RandomGenarator:
    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)


class FarField1DSource(UniformLinearSensorArray, RandomGenarator):
    __slots__ = ['num_sample', 'num_target', 'coherent', 'is_baseband', 'sampling_time']

    def __init__(
        self,
        num_sample: int,
        num_target: int,
        coherent: bool,
        num_antenna: int,
        freq: int,
        is_baseband: bool = False,
        element_spacing: float = 0.5,  # means lambda / 2
        angle_type: str = 'rad',
        **kwargs
    ):
        super().__init__(num_antenna, freq, element_spacing, angle_type, **kwargs)
        if num_sample <= 0:
            raise ValueError("Number of samples must be positive.")
        if num_target <= 0:
            raise ValueError("Number of targets must be positive.")

        self.num_sample = num_sample
        self.num_target = num_target
        self.coherent = coherent
        self.is_baseband = is_baseband

        if not is_baseband:
            self.freq_sampling = self.freq
            self.sampling_time = self._compute_sampling_time(self.freq)

    def _compute_sampling_time(self, freq) -> np.ndarray:
        freq_sampling = 4 * freq
        return np.arange(self.num_sample) / freq_sampling

    def _emitted_normal_signal(self) -> np.ndarray:
        """
        output shape:
            (num_target, num_sample)
        """
        if self.coherent:
            sig = (
                self.rng.standard_normal((1, self.num_sample)) +
                1j * self.rng.standard_normal((1, self.num_sample)))
            amplitudes = self.rng.uniform(0.5, 1.5, (self.num_target, 1))
            sig = sig * amplitudes
        else:
            sig = (
                self.rng.standard_normal((self.num_target, self.num_sample)) +
                1j * self.rng.standard_normal((self.num_target, self.num_sample))
            )

        return sig

    def _emitted_sinusoidal_signal(self) -> np.ndarray:
        """
        output shape:
            (num_target, num_sample)
        """
        t = self.sampling_time[np.newaxis, :]

        if self.coherent:
            random_amplitudes = self.rng.uniform(0.5, 1.5, (self.num_target, 1))
            phase = 2 * np.pi * self.freq_sampling * t
            sig = np.exp(-1j * phase) * random_amplitudes
        else:
            random_phases = self.rng.uniform(0, 2*np.pi, self.num_target)
            random_amplitudes = self.rng.uniform(0.5, 1.5, self.num_target)
            phase = 2 * np.pi * self.freq_sampling * t + random_phases[:, np.newaxis]
            sig = np.exp(-1j * phase) * random_amplitudes[:, np.newaxis]
        return sig

    def _complex_normal_noise(self) -> np.ndarray:
        """
        output shape:
            (num_sample, num_antenna)
        """
        n = (self.rng.standard_normal((self.num_sample, self.num_antenna)) +
             1j * self.rng.standard_normal((self.num_sample, self.num_antenna)))
        return n / np.sqrt(2)

    def collect_plane_wave_response(self, angles: np.ndarray, snr: int, num_antenna: int = None) -> np.ndarray:
        """
        output shape:
            (num_sample, num_antenna)
        """
        if angles.shape[0] != self.num_target:
            raise ValueError("Number of angles must match the number of targets.")

        if self.is_baseband:
            S = self._emitted_normal_signal()
        else:
            S = self._emitted_sinusoidal_signal()
        N = self._complex_normal_noise()
        A = self.steering_matrix(angles, num_antenna)

        X = A @ S
        sig_p = np.mean(np.abs(X)**2)
        scaling = np.sqrt(10 ** (snr * 0.1) / sig_p)
        X *= scaling

        return X.T + N

    def collect_plane_wave_response_doublets(self, angles: np.ndarray, snr: int, num_antenna: int = None) -> np.ndarray:
        """
        output shape:
            (num_sample, num_antenna)
        """
        if angles.shape[0] != self.num_target:
            raise ValueError("Number of angles must match the number of targets.")

        if self.is_baseband:
            S = self._emitted_normal_signal()
        else:
            S = self._emitted_sinusoidal_signal()
        N = self._complex_normal_noise()
        A = self.steering_matrix(angles, num_antenna)
        phi = self.doublet_phase_delays_matrix(angles)

        X = (A @ phi) @ S
        sig_p = np.mean(np.abs(X)**2)
        scaling = np.sqrt(10 ** (snr * 0.1) / sig_p)
        X *= scaling

        return X.T + N
