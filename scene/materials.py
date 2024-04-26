class Material:
    def __init__(self) -> None:
        pass

    def getColor(self):
        return (0, 0, 0)


class BaseMaterial(Material):
    def __init__(self, color) -> None:
        self.color = color

    def getColor(self):
        return self.color
