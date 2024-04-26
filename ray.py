import numpy as np
import math


class Ray:
    distance_traveled = 0
    velocity = np.ndarray
    position = np.ndarray

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
        try:
            a = math.pow(math.dist(self.velocity, np.zeros(len(self.velocity))), -1.0)
        except:
            a = 0

        b = np.multiply(self.velocity, a)
        deltaPos = np.multiply(b, distance)

        self.position = np.add(self.position, deltaPos)
        self.distance_traveled += distance
