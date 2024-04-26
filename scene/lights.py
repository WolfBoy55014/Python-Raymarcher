from ray import Ray
import numpy as np
from util import normalize


class SceneLight:

    def __init__(self) -> None:
        pass

    def getPosition(self):
        pass

    def getIntensity(self):
        pass

    def getColor(self):
        pass

    def getLightVector(self, ray: Ray):
        pass


class PointLight:
    def __init__(self, pos: np.ndarray, intensity: float, color: np.ndarray) -> None:
        self.pos = pos
        self.intensity = intensity
        self.color = color

    def getPosition(self):
        return self.pos

    def getIntensity(self):
        return self.intensity

    def getColor(self):
        return self.color

    def getLightVector(self, ray: Ray):
        return normalize(self.pos - ray.getPosition())
