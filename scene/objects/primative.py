import math
import numpy as np
from ray import Ray
from scene.materials import Material
from scene.objects.scene_object import SceneObject
from util import dist
from numba import jit


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

        return Sphere._sdf(ray.getPosition(), self.pos, self.radius)

    @jit(cache=True)
    def _sdf(ray_position: np.ndarray, pos: np.ndarray, radius: float):
        return dist(ray_position, pos) - radius

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

        return Torus._sdf(
            ray.getPosition(), self.pos, self.major_radius, self.minor_radius
        )

    def _sdf(
        ray_position: np.ndarray,
        pos: np.ndarray,
        major_radius: float,
        minor_radius: float,
    ):
        # Translate the point by the negative center to effectively move the torus
        p_relative = np.subtract(ray_position, pos)

        q = (
            dist((p_relative[0], p_relative[1]), (0, 0)) - major_radius,
            p_relative[2],
        )
        return dist(q, (0, 0)) - minor_radius

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

        return Cylinder._sdf(ray.getPosition(), self.pos, self.height, self.radius)

    @jit(cache=True)
    def _sdf(ray_position: np.ndarray, pos: np.ndarray, height: float, radius: float):
        # Translate the point by the negative center to effectively move the cylinder
        p_relative = np.subtract(np.asarray(ray_position), np.asarray(pos))

        d = dist((p_relative[0], p_relative[1]), (0, 0)) - radius
        r = max(-(p_relative[2] + height / 2), d)
        f = max((p_relative[2] - height / 2), r)

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

        return Cube._sdf(ray.getPosition(), self.pos, self.side_length)

    @jit(cache=True)
    def _sdf(ray_position: np.ndarray, pos: np.ndarray, side_length: float):
        a = side_length / 2

        # Translate the point by the negative center to effectively move the cube
        p_relative = np.subtract(np.asarray(ray_position), np.asarray(pos))
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

        return Box._sdf(ray.getPosition(), self.pos, self.side_lengths)

    @jit(cache=True)
    def _sdf(ray_position: np.ndarray, pos: np.ndarray, side_lengths: np.ndarray):
        a = np.divide(np.asarray(side_lengths), np.full(3, 2, np.float64))

        # Translate the point by the negative center to effectively move the box
        p_relative = np.subtract(np.asarray(ray_position), np.asarray(pos))
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

        return RoundBox._sdf(
            ray.getPosition(), self.pos, self.side_lengths, self.radius
        )

    @jit(cache=True)
    def _sdf(
        ray_position: np.ndarray,
        pos: np.ndarray,
        side_lengths: np.ndarray,
        radius: float,
    ):
        a = np.divide(np.asarray(side_lengths), np.full(3, 2, np.float64))

        # Translate the point by the negative center to effectively move the box
        p_relative = np.subtract(np.asarray(ray_position), np.asarray(pos))
        d = np.abs(p_relative) - a  # Distance to each face along each axis
        return np.max(d) - radius  # Choose the maximum distance for the closest face

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

        return Plane._sdf(ray.getPosition(), self.pos, self.axis)
    
    # @jit(cache=True)     
    def _sdf(ray_position: np.ndarray, pos: float, axis: str):
        if axis == "Z":
            return (
                abs(ray_position[2] - pos) - 0.02
            )
        elif axis == "Y":
            return (
                abs(ray_position[1] - pos) - 0.02
            )
        elif axis == "X":
            return (
                abs(ray_position[0] - pos) - 0.02
            )
        else:
            print('Invalid Axis, try "X", "Y", or "Z".')

    def getMaterial(self):
        return self.material
