import numpy as np


class ToneMapping:

    # Calculates the luminance of a color
    def luminance(color: np.ndarray):
        return np.dot(color, (0.2126, 0.7152, 0.0722))

    # Adjusts a color to have a desired luminance
    def change_luminance(color: np.ndarray, desired_luminance: float):
        l_in = ToneMapping.luminance(color)
        return np.multiply(color, (desired_luminance / l_in))

    def extendedReinhard(color: np.ndarray):
        color = np.divide(color, 255)

        # print("Before: " + str(color))

        luminance = ToneMapping.luminance(color)

        new_luminance = (((luminance / 4) + 1) * luminance) / (1 + luminance)

        color = ToneMapping.change_luminance(color, new_luminance)

        # print("After:  " + str(color))

        color = np.multiply(color, 255)

        return color
