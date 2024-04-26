from ray import Ray
import numpy as np
from scene.objects.scene_object import SceneObject

class UnionObject(SceneObject):

    def __init__(self, object1, object2) -> None:
        self.object1 = object1
        self.object2 = object2
        self.nearest_object = None

    def getSDF(self, ray: Ray) -> float:
        distances = (self.object1.getSDF(ray), self.object2.getSDF(ray))

        # Save which object was colided with, so we can use its material
        if np.argmin(distances) == 0:
            self.nearest_object = self.object1
        else:
            self.nearest_object = self.object2

        return min(distances)

    def getMaterial(self):
        return self.nearest_object.getMaterial()


class IntersectionObject(SceneObject):

    def __init__(self, object1, object2) -> None:
        self.object1 = object1
        self.object2 = object2
        self.nearest_object = None

    def getSDF(self, ray: Ray) -> float:
        distances = (self.object1.getSDF(ray), self.object2.getSDF(ray))

        # Save which object was colided with, so we can use its material
        if np.argmax(distances) == 0:
            self.nearest_object = self.object1
        else:
            self.nearest_object = self.object2

        return max(distances)

    def getMaterial(self):
        return self.nearest_object.getMaterial()


class DifferenceObject(SceneObject):

    def __init__(self, object1, object2) -> None:
        self.object1 = object1
        self.object2 = object2
        self.nearest_object = None

    def getSDF(self, ray: Ray) -> float:
        distances = (-self.object1.getSDF(ray), self.object2.getSDF(ray))

        # Save which object was colided with, so we can use its material
        if np.argmax(distances) == 0:
            self.nearest_object = self.object1
        else:
            self.nearest_object = self.object2

        return max(distances)

    def getMaterial(self):
        return self.nearest_object.getMaterial()


class ScaledObject(SceneObject):

    def __init__(self, object1, scale: np.ndarray) -> None:
        self.object1 = object1
        self.scale = [1 / x for x in scale]

    def getSDF(self, ray: Ray) -> float:

        # For scaling to work, the translation must happen first,
        # Sadly translation is handled by this object's child.
        # So, we must add the object's position, so it gets cancled out after passing through the child

        p_old = ray.getPosition()

        p_relative = np.subtract(ray.getPosition(), self.object1.getPos())
        p_scaled = np.multiply(p_relative, self.scale)

        # Cancel out the child's internal translation
        p = np.add(p_scaled, self.object1.getPos())
        ray.setPosition(p)

        distance = self.object1.getSDF(ray)

        # Reset ray to oeiginal position
        ray.setPosition(p_old)

        return distance

    def getMaterial(self):
        return self.object1.getMaterial()
