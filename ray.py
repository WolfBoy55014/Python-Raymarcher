import numpy as np
import math
from numba import jit


class Ray:
    distance_traveled: float = 0
    velocity = np.zeros(3)
    position = np.zeros(3)

    def __init__(self, velocity: np.ndarray, position: np.ndarray) -> None:
        self.velocity = self._normalize(velocity)
        self.position = position

    def _normalize(self, array: np.ndarray):
        if np.max(np.abs(array)) == 0:
            return np.zeros(len(array))

        magnitude = math.dist(array, (0, 0, 0))
        normalized_array = np.divide(array, magnitude)

        return normalized_array

    def getVelocity(self):
        return self.velocity

    def setVelocity(self, velocity: np.ndarray):
        self.velocity = velocity

    def getPosition(self):
        return self.position

    def setPosition(self, pos: np.ndarray):
        self.position = pos

    def resetDistance(self):
        self.distance_traveled = 0

    def getX(self):
        return self.position[0]

    def getY(self):
        return self.position[1]

    def getZ(self):
        return self.position[2]

    def step(self, distance):
        self.position, self.distance_traveled = Ray._step(
            self.velocity, self.position, distance, self.distance_traveled
        )

    @jit(cache=True)
    def _step(velocity, position, distance, distance_traveled):
        try:
            a = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)
        except:
            a = 0

        b = np.divide(velocity, a)
        deltaPos = np.multiply(b, distance)

        new_position = np.asarray(position, np.float32) + np.asarray(
            deltaPos, np.float32
        )
        distance_traveled = distance_traveled + distance

        return new_position, distance_traveled
