import numpy as np
from scipy import linalg
from signal_model.antenna_response import FarField1DSource


class Capon(FarField1DSource):
    def __init__(self, all_doas, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_doas = all_doas

    def _calc_weights(self, R_inv, stv):
        w = R_inv @ stv
        w /= (stv.conj().T @ R_inv @ stv)
        return w

    def _calc_power(self, weights, signal):
        y = np.dot(weights.conj().T, signal.T)
        power = np.mean(np.abs(y)**2)
        return power

    def estimate(self, doas: np.ndarray, snr: int):
        power = np.empty_like(self.all_doas, dtype=np.float32)
        sig = self.collect_plane_wave_response(doas, snr)
        R = np.cov(sig, rowvar=False)
        try:
            R_inv = linalg.inv(R)
        except linalg.LinAlgError:
            raise ValueError("Failed to invert R.")

        for i, doa in enumerate(self.all_doas):
            stv = self.steering_vector(doa)[:, np.newaxis]

            w = self._calc_weights(R_inv, stv)
            power[i] = self._calc_power(w, sig)

        return power
