from ray import Ray
from util import normalize
import numpy as np


class SceneObject:
    """Defines a generic primitive scene object"""

    def __init__(self) -> None:
        pass

    def getPos(self):
        pass

    def getSDF(self, ray: Ray) -> float:
        pass

    def getMaterial(self):
        pass

    def getNormal(self, ray: Ray):
        d = self.getSDF(ray)
        min_distance = 0.01

        x_normal = d - self.getSDF(
            Ray((0, 0, 0), ray.getPosition() - (min_distance, 0.0, 0.0))
        )
        y_normal = d - self.getSDF(
            Ray((0, 0, 0), ray.getPosition() - (0.0, min_distance, 0.0))
        )
        z_normal = d - self.getSDF(
            Ray((0, 0, 0), ray.getPosition() - (0.0, 0.0, min_distance))
        )

        normal = np.zeros(3)
        normal[0] = x_normal
        normal[1] = y_normal
        normal[2] = z_normal
        return normalize(normal)
