import numpy as np


class UniformLinearSensorArray:
    __slots__ = ['d', 'num_antenna', 'freq', '_lambda', '_is_degrees']

    def __init__(
        self,
        num_antenna: int,
        freq: int,
        element_spacing: float = 0.5,  # means lambda / 2
        angle_type: str = 'rad'
    ):
        super().__init__()
        if angle_type.lower() not in ['rad', 'deg']:
            raise ValueError(
                f"Invalid angle_type '{angle_type}'. Supported types are 'deg' and 'rad'.")
        if freq <= 0:
            raise ValueError("Frequency must be positive.")
        if num_antenna <= 0:
            raise ValueError("Number of antennas must be positive.")
        if element_spacing <= 0:
            raise ValueError("Element spacing must be positive.")

        self._is_degrees = angle_type == 'deg'
        self.freq = freq
        self.num_antenna = num_antenna
        self._lambda = 3e8 / self.freq
        self.d = element_spacing * self._lambda

    def steering_vector(self, angle: float, num_antenna: int = None) -> np.ndarray:
        """
        output shape:
            (num_antenna,)
        """
        if self._is_degrees:
            angle = np.deg2rad(angle)

        if num_antenna is None:
            num_antenna = self.num_antenna

        element_positions = np.arange(self.num_antenna) * self.d * np.sin(angle) / self._lambda
        return np.exp(-2j * np.pi * element_positions)

    def steering_matrix(self, angles: np.ndarray, num_antenna: int = None) -> np.ndarray:
        """
        output shape:
            (num_antenna, num_angles)
        """
        if self._is_degrees:
            angles = np.deg2rad(angles)

        if num_antenna is None:
            num_antenna = self.num_antenna

        num_angles = angles.shape[0]
        element_indices = np.arange(num_antenna)[:, np.newaxis]
        angles_reshaped = np.reshape(angles, (1, num_angles))
        element_positions = element_indices * self.d * np.sin(angles_reshaped) / self._lambda
        return np.exp(-2j * np.pi * element_positions)

    def steering_matrix_derivative(self, angles: np.ndarray) -> np.ndarray:
        """
        output shape:
            (num_antenna, num_angles)
        """
        if self._is_degrees:
            angles = np.deg2rad(angles)
        num_angles = angles.shape[0]

        angles_reshaped = np.reshape(angles, (1, num_angles))
        cos_angles = np.cos(angles_reshaped)
        n = np.arange(self.num_antenna)[:, np.newaxis]
        st_v = np.exp(-2j * np.pi * n * self.d * np.sin(angles_reshaped) / self._lambda)

        return -2j * np.pi * n * self.d * cos_angles * st_v / self._lambda

    def doublet_phase_delays_matrix(self, angles: np.ndarray) -> np.ndarray:
        """
        output shape:
            (num_angles, num_angles)
        """
        if self._is_degrees:
            angles = np.deg2rad(angles)

        num_angles = angles.shape[0]
        phi = np.zeros((num_angles, num_angles), dtype=np.complex128)
        for i, angle in enumerate(angles):
            phi[i, i] = np.exp(1j * np.sin(angle))

        return phi
