import numpy as np


def generate_random_angles(num_targets, all_angles_range, min_separation) -> np.ndarray:
    angles = np.empty(num_targets)
    available_mask = np.ones(len(all_angles_range), dtype=bool)
    available_indices = np.arange(len(all_angles_range))

    for i in range(num_targets):
        valid_indices = available_indices[available_mask]

        if len(valid_indices) == 0:
            return angles[:i]

        chosen_idx = np.random.choice(valid_indices)
        selected_angle = all_angles_range[chosen_idx]
        angles[i] = selected_angle

        angle_diffs = np.abs(all_angles_range - selected_angle)
        available_mask &= (angle_diffs >= min_separation)

    return angles
