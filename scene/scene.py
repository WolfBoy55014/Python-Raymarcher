from ray import Ray
import numpy as np
import math
from util import normalize, clamp

class Scene:

    def __init__(
        self,
        objects: np.ndarray,
        lights: np.ndarray,
        min_distance: float,
        max_distance: float,
        do_shading: bool,
    ) -> None:
        self.objects = objects
        self.lights = lights
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.do_shading = do_shading
        self.distances = []
        self.normal = [0, 0, 0]

    def getSDF(self, ray: Ray) -> float:
        self.distances = []

        for object in self.objects:
            self.distances.append(object.getSDF(ray))

        return min(self.distances)

    def getNormal(self, ray: Ray):
        normal = self.getNearestObject(ray).getNormal(ray)
        self.normal = normal
        return normal
        
    def getNearestObject(self, ray: Ray):
        # Base Color of object
        index = np.argmin(self.distances)  # Get Index of nearest object
        return self.objects[index]  # Get nearest object
        

    def getColor(self, ray: Ray):

        nearest_object = self.getNearestObject(ray)

        # Set the base color of the pixel to the nearest objects material color
        object_color = nearest_object.getMaterial().getColor()

        color = (17, 17, 17)  # Start with a blank (or black) color

        # Calculate the scene's normal at the ray's position
        normal = self.getNormal(ray)

        # Shading
        for light in self.lights:
            if not self.do_shading:
                color = object_color
                break

            # Diffused Lighting
            brightness = np.dot(light.getLightVector(ray), normal)

            # Clamp brightness between 0 and 1, because negative brightness does not exist!
            brightness = clamp(brightness, 0, 1)

            # Apply shadows
            brightness *= self.isInShadow(ray, light, normal, 16)

            # Multiply the brightness by the light's intensity to allow for dimming
            brightness *= light.getIntensity()

            # Get the light's color
            light_color = light.getColor()

            # Now we are going to compine the light_color and object_color into one by multiplying
            # This works best when at least one is converted to 0 - 1.
            light_color = np.divide(light_color, 255)

            # object_color * light_color * brightness
            color += np.multiply(np.multiply(object_color, light_color), brightness)

        return color

    def isInShadow(self, ray: Ray, light, normal, softness):
        starting_pos = ray.getPosition()
        starting_velocity = ray.getVelocity()

        # We need to move the ray away from the surface a bit to not detect a false hit
        ray.setPosition(ray.getPosition() + (normal * self.min_distance * 2))

        # Move the ray to face the light
        ray.setVelocity(light.getLightVector(ray))
        ray.resetDistance()

        # To figure out if we are in a shadow, we march towards the light,
        # if the distance we traveled is shorter than our distance from the light,
        # we are in a shadow
        distance = lambda ray: self.getSDF(ray)

        brightness = 1.0

        d = distance(ray)
        while (d > self.min_distance) and (ray.distance_traveled < self.max_distance):
            d = distance(ray)

            # Step smaller steps as we get closer to the scene to improve pnumara quality
            # move = min(.2 * math.pow(d, 2), d)
            # d = max(0.2 * d, d - 5) - 2
            # d *= 0.7 # Working

            # min(.0x^2+0.1, x) - 1

            if d <= 0.5:
                ray.step(d * 0.5)
            else:
                ray.step(d)

            brightness = min((d / ray.distance_traveled) * softness, brightness)

        # The distance we were from the light when we started
        starting_distance = math.dist(starting_pos, light.getPosition())

        # Put the ray back where it was
        ray.setPosition(starting_pos)
        ray.setVelocity(starting_velocity)

        if ray.distance_traveled >= starting_distance:
            # We were not in a shadow
            return brightness
        else:
            # We were in a shadow
            return 0.0
