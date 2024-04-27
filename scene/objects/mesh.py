from ray import Ray
from scene.objects.scene_object import SceneObject
import numpy as np
from scene.materials import *
from util import clamp
import trimesh
from trimesh.proximity import ProximityQuery
import math


class MeshObject(SceneObject):

    def __init__(self, pos: np.ndarray, scale: np.ndarray, file_name: str, material: Material) -> None:
        self.pos = (pos[0], pos[1], pos[2])
        self.material = material
        self.mesh: trimesh.Trimesh = trimesh.load(file_name, force="mesh")

        # self.mesh.convert_units("meters", True)
        self.mesh.apply_scale(scale)
        self.mesh.apply_translation(self.pos)

        print("Mesh at:", self.mesh.centroid)
        print("Mesh vertices:", self.mesh.vertices)

        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.normals = self.mesh.face_normals

    def getPos(self):
        return self.pos

    def getSDF(self, ray: Ray) -> float:
        distances = []

        for face in self.faces:

            distances.append(
                self._triangle(
                    ray,
                    self.vertices[face[0]],
                    self.vertices[face[1]],
                    -self.vertices[face[2]],
                )
            )

        return math.sqrt(min(distances)) - 0.05

    def _triangle(self, ray: Ray, a, b, c) -> float:
        p = ray.getPosition()
        # vec3 ba = b - a; vec3 pa = p - a;
        ba = b - a
        pa = p - a
        # vec3 cb = c - b; vec3 pb = p - b;
        cb = c - b
        pb = p - b
        # vec3 ac = a - c; vec3 pc = p - c;
        ac = a - c
        pc = p - c
        # vec3 nor = cross( ba, ac );
        nor = np.cross(ba, ac)

        d = (
            MeshObject._sign(np.dot(np.cross(ba, nor), pa))
            + MeshObject._sign(np.dot(np.cross(cb, nor), pb))
            + MeshObject._sign(np.dot(np.cross(ac, nor), pc))
        )

        e = min(
            min(
                MeshObject._dot2(
                    ba * clamp(np.dot(ba, pa) / MeshObject._dot2(ba), 0.0, 1.0) - pa
                ),
                MeshObject._dot2(
                    cb * clamp(np.dot(cb, pb) / MeshObject._dot2(cb), 0.0, 1.0) - pb
                ),
            ),
            MeshObject._dot2(
                ac * clamp(np.dot(ac, pc) / MeshObject._dot2(ac), 0.0, 1.0) - pc
            ),
        )

        f = np.dot(nor, pa) * np.dot(nor, pa) / MeshObject._dot2(nor)

        return e if d < 2.0 else f

    # float dot2( in vec3 v ) { return dot(v,v); }
    def _dot2(v):
        return np.dot(v, v)

    def _sign(value):
        if value < 0:
            return -1
        elif value > 0:
            return 1
        else:
            return 0

    def getMaterial(self):
        return self.material

    def getNormal(self, ray: Ray):
        return super().getNormal(ray)
