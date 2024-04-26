import numpy as np
import math

def normalize(array: np.ndarray):

    if np.max(np.abs(array)) == 0:
        return np.zeros(len(array))

    magnitude = math.dist(array, (0, 0, 0))
    normalized_array = np.divide(array, magnitude)

    return normalized_array


def get_initial_velocity(x, y, image_width, image_height, fov, camera_rotation):
    return np.add(
        (
            cast(x, 0, image_width * fov, -1, 1),
            1.0,
            cast(y, 0, image_height * fov, -0.5625, 0.5625),
        ),
        camera_rotation,
    )


def cast(value, old_min, old_max, new_min, new_max):
    return (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


def clamp(value, min, max):
    if value > max:
        return max
    elif value < min:
        return min
    return value
