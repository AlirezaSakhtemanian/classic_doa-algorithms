import numpy as np
from scipy import linalg
from signal_model.sensor_array import UniformLinearSensorArray


class CramerRaoBound(UniformLinearSensorArray):
    def __init__(self, num_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

    def crb_stochastic(self, doas, snr):
        doas = np.array(doas).flatten()
        num_doas = len(doas)

        A = self.steering_matrix(doas)
        dA = self.steering_matrix_derivative(doas)

        snr_linear = 10**(snr/10)

        P = np.eye(num_doas) * snr_linear

        R = A @ P @ A.conj().T + np.eye(self.num_antenna)
        R_inv = linalg.inv(R)

        FIM = np.zeros((num_doas, num_doas), dtype=complex)

        for i in range(num_doas):
            for j in range(num_doas):
                dA_i = dA[:, i].reshape(-1, 1)
                dA_j = dA[:, j].reshape(-1, 1)

                term1 = np.trace(R_inv @ dA_i @ P[i:i+1, :] @ A.conj().T @ R_inv @ A @ P[:, j:j+1] @ dA_j.conj().T)
                term2 = np.trace(R_inv @ dA_i @ P[i:i+1, j:j+1] @ dA_j.conj().T)
                term3 = np.trace(R_inv @ A @ P[:, j:j+1] @ dA_j.conj().T @ R_inv @ dA_i @ P[i:i+1, :] @ A.conj().T)
                term4 = np.trace(R_inv @ A @ P @ A.conj().T @ R_inv @ dA_i @ P[i:i+1, j:j+1] @ dA_j.conj().T)

                FIM[i, j] = self.num_samples * np.real(term1 + term2 + term3 + term4)

        try:
            if num_doas == 1:
                crb = 1 / FIM[0, 0]
            else:
                crb = np.diag(linalg.inv(FIM))
            return crb
        except np.linalg.LinAlgError:
            print("Warning: Fisher Information Matrix is singular or poorly conditioned")
            return np.full(num_doas, np.inf)
