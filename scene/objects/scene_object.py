from ray import Ray

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
