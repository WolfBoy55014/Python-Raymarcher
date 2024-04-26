import math
import numpy as np
from ray import Ray
from scene.materials import Material
from scene.objects.scene_object import SceneObject


class Sphere(SceneObject):
    """Defines sphere scene object"""

    def __init__(self, pos: np.ndarray, radius: float, material: Material) -> None:
        self.pos = pos
        self.radius = radius
        self.material = material

    def getPos(self):
        return self.pos

    def getSDF(self, ray: Ray) -> float:
        """
        Calculates the signed distance to a sphere with radius and center position

        ## Args:
            `ray`: A ray to calculate the distance from.
            `pos`: A 3D vector representing the center position of the sphere.
            `radius`: Radius of the sphere.

        ## Returns:
            The signed distance to the sphere's surface from the ray.
        """

        return math.dist(ray.getPosition(), self.pos) - self.radius

    def getMaterial(self):
        return self.material


class Torus(SceneObject):
    """Defines torus scene object"""

    def __init__(
        self,
        pos: np.ndarray,
        major_radius: float,
        minor_radius: float,
        material: Material,
    ) -> None:
        self.pos = pos
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.material = material

    def getPos(self):
        return self.pos

    def getSDF(self, ray: Ray) -> float:
        """
        Calculates the signed distance to a torus with a radius, minor radius and center position

        ## Args:
            `ray`: A ray to calculate the distance from.
            `pos`: A 3D vector representing the center position of the sphere.
            `major_radius`: Radius of the torus.
            `minor_radius`: Minor radius or thickness of the torus.

        ## Returns:
            The signed distance to the torus's surface from the ray.
        """

        # Translate the point by the negative center to effectively move the torus
        p_relative = np.subtract(ray.getPosition(), self.pos)

        q = (
            math.dist((p_relative[0], p_relative[1]), (0, 0)) - self.major_radius,
            p_relative[2],
        )
        return math.dist(q, (0, 0)) - self.minor_radius

    def getMaterial(self):
        return self.material


class Cylinder(SceneObject):
    """Defines cylinder scene object"""

    def __init__(
        self, pos: np.ndarray, height: float, radius: float, material: Material
    ) -> None:
        self.pos = pos
        self.radius = radius
        self.height = height
        self.material = material

    def getPos(self):
        return self.pos

    def getSDF(self, ray: Ray) -> float:
        """
        Calculates the signed distance to a torus with a radius, minor radius and center position

        ## Args:
            `ray`: A ray to calculate the distance from.
            `pos`: A 3D vector representing the center position of the sphere.
            `height`: Height or length of the cylinder.
            `radius`: Radius of the cylinder.

        ## Returns:
            The signed distance to the cylinder's surface from the ray.
        """

        # Translate the point by the negative center to effectively move the cylinder
        p_relative = np.subtract(ray.getPosition(), self.pos)

        d = math.dist((p_relative[0], p_relative[1]), (0, 0)) - self.radius
        r = max(-(p_relative[2] + self.height / 2), d)
        f = max((p_relative[2] - self.height / 2), r)

        return f

    def getMaterial(self):
        return self.material


class Cube(SceneObject):
    """Defines cube scene object"""

    def __init__(self, pos: np.ndarray, side_length: float, material: Material) -> None:
        self.pos = pos
        self.side_length = side_length
        self.material = material

    def getPos(self):
        return self.pos

    def getSDF(self, ray: Ray) -> float:
        """
        Calculates the signed distance to a cube with side length 2a and center position.

        ## Args:
            `ray`: A ray to calculate the distance from.
            `pos`: A 3D vector representing the center position of the cube.
            `side_length`: Half the side length of the cube.

        ## Returns:
            The signed distance to the cube's surface from the ray.
        """

        a = self.side_length / 2
        ray_position = ray.getPosition()

        # Translate the point by the negative center to effectively move the cube
        p_relative = np.subtract(ray_position, self.pos)
        d = np.abs(p_relative) - a  # Distance to each face along each axis
        return np.max(d)  # Choose the maximum distance for the closest face

    def getMaterial(self):
        return self.material


class Box(SceneObject):
    """Defines box scene object"""

    def __init__(
        self, pos: np.ndarray, side_lengths: np.ndarray, material: Material
    ) -> None:
        self.pos = pos
        self.side_lengths = side_lengths
        self.material = material

    def getPos(self):
        return self.pos

    def getSDF(self, ray: Ray) -> float:
        """
        Calculates the signed distance to a cube with side length 2a and center position.

        ## Args:
            `ray`: A ray to calculate the distance from.
            `pos`: A 3D vector representing the center position of the cube.
            `side_lengths`: A list containing the side length of the box

        ## Returns:
            The signed distance to the boxes's surface from the ray.
        """

        a = np.divide(self.side_lengths, 2)
        ray_position = ray.getPosition()

        # Translate the point by the negative center to effectively move the box
        p_relative = np.subtract(ray_position, self.pos)
        d = np.abs(p_relative) - a  # Distance to each face along each axis
        return np.max(d)  # Choose the maximum distance for the closest face

    def getMaterial(self):
        return self.material


class RoundBox(SceneObject):
    """Defines box scene object"""

    def __init__(
        self,
        pos: np.ndarray,
        side_lengths: np.ndarray,
        radius: float,
        material: Material,
    ) -> None:
        self.pos = pos
        self.side_lengths = side_lengths
        self.radius = radius
        self.material = material

    def getPos(self):
        return self.pos

    def getSDF(self, ray: Ray) -> float:
        """
        Calculates the signed distance to a cube with side length 2a and center position.

        ## Args:
            `ray`: A ray to calculate the distance from.
            `pos`: A 3D vector representing the center position of the cube.
            `side_lengths`: A list containing the side length of the box

        ## Returns:
            The signed distance to the boxes's surface from the ray.
        """

        a = np.divide(self.side_lengths, 2)
        ray_position = ray.getPosition()

        # Translate the point by the negative center to effectively move the box
        p_relative = np.subtract(ray_position, self.pos)
        d = np.abs(p_relative) - a  # Distance to each face along each axis
        return (
            np.max(d) - self.radius
        )  # Choose the maximum distance for the closest face

    def getMaterial(self):
        return self.material


class Plane(SceneObject):
    """Defines axis-aligned plane scene object"""

    def __init__(self, axis: str, pos: float, material: Material) -> None:
        self.axis = axis
        self.pos = pos
        self.material = material

    def getPos(self):
        return self.pos

    def getSDF(self, ray: Ray) -> float:
        """
        Calculates the signed distance to a horizontal plane at height z.

        ## Args:
            `ray`: A ray to calculate the distance from.
            `z`: The plane's height or location on the z axis.

        ## Returns:
            The signed distance to the plane's surface from the ray.
        """

        if self.axis == "Z":
            return (
                abs(math.dist(ray.getPosition(), (ray.getX(), ray.getY(), self.pos)))
                - 0.02
            )
        elif self.axis == "Y":
            return (
                abs(math.dist(ray.getPosition(), (ray.getX(), self.pos, ray.getZ())))
                - 0.02
            )
        elif self.axis == "X":
            return (
                abs(math.dist(ray.getPosition(), (self.pos, ray.getY(), ray.getZ())))
                - 0.02
            )
        else:
            print('Invalid Axis, try "X", "Y", or "Z".')

    def getMaterial(self):
        return self.material
