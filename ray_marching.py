import os, sys

print(os.getcwd())
sys.path.insert(0, os.getcwd())

# Imports
import numpy as np
from PIL import Image
from tqdm import tqdm
from ray import Ray
from scene.scene import Scene
from scene.materials import BaseMaterial
from scene.objects.primative import *
from scene.objects.mesh import *
from scene.objects.modifier import *
from scene.lights import PointLight
from processing import ToneMapping
from util import get_initial_velocity
import time
from multiprocessing import Pool, freeze_support, set_start_method
from functools import partial

# Constants

# Image Size
# 72p = 128 * 72 (Best)
# 144p = 256 x 144 pixels
# 240p = 426 x 240 pixels
# 720p = 1280 x 720 pixels
# 1080p = 1920 x 1080 pixels
image_height = 144
image_width = 256
image_size = (image_width, image_height)

contrast = 70
fov = 1
shading = False

camera_pos = (0, -1.5, -1)
camera_rotation = (0, 0, 0.5)

min_distance = 0.001
max_distance = 25


def hit(scene: Scene, ray: Ray):
    color = scene.getColor(ray)

    # Tone Mapping
    color = ToneMapping.extendedReinhard(color)

    return color


def miss(scene: Scene, ray: Ray):
    return (0, 0, 0)


def render(x, y, scene):
    # Get the velocity vector for the ray
    velocity = get_initial_velocity(
        x, y, image_width, image_height, fov, camera_rotation
    )

    # Create Ray
    ray = Ray(velocity, camera_pos)

    # This is the real interesting part,
    # This grabs the distance from the ray to the scene,
    distance = lambda ray: scene.getSDF(ray)
    # Then, if that distance is larger than our set minimum distance,
    d = distance(ray)
    while (d > min_distance) and (ray.distance_traveled < max_distance):
        d = distance(ray)

        # we move by that distance.
        ray.step(d)

        if d <= min_distance:
            return hit(scene, ray)

        elif ray.distance_traveled >= max_distance:
            return miss(scene, ray)

    return miss(scene, ray)

def renderRow(scene, y):
    row = []
    for x in range(image_width):
        row.append(render(x, y, scene))
    return row

if __name__ == "__main__":
    # Enable Multiprocessing Support
    freeze_support()
    set_start_method('spawn')
    
    
    color1 = (77, 32, 21)
    color2 = (177, 103, 57)
    color3 = (195, 178, 159)
    color4 = (229, 169, 59)
    color5 = (4, 111, 147)
    color6 = (83, 149, 67)

    # Defining Material
    dark_wood_mat = BaseMaterial(color1)
    light_wood_mat = BaseMaterial(color3)
    orange_mat = BaseMaterial(color2)
    yellow_mat = BaseMaterial(color4)
    blue_mat = BaseMaterial(color5)
    green_mat = BaseMaterial(color6)


    # Defining Objects
    desk_top1 = Box((0, 0, -0.6), (2, 0.5, 0.05), dark_wood_mat)
    desk_top2 = Box((0, 0, -0.55), (2, 0.5, 0.05), dark_wood_mat)
    desk_leg1 = Box((-0.9, 0, -0.26), (0.05, 0.4, 0.55), dark_wood_mat)
    desk_leg2 = Box((-0.4, 0, -0.26), (0.05, 0.4, 0.55), dark_wood_mat)
    desk_leg3 = Box((0.9, 0, -0.26), (0.05, 0.4, 0.55), dark_wood_mat)
    drawer1 = Box((-0.65, 0, -0.425), (0.45, 0.37, 0.2), light_wood_mat)
    drawer2 = Box((-0.65, 0, -0.175), (0.45, 0.37, 0.2), light_wood_mat)
    drawer_mid = Box((-0.65, 0, -0.3), (0.45, 0.4, 0.05), dark_wood_mat)
    drawer_bottom = Box((-0.65, 0, -0.05), (0.45, 0.4, 0.05), dark_wood_mat)

    ground = Plane("Z", 0, blue_mat)

    # Defining Lights
    side_light1 = PointLight((-1, -1, -1), 2, (255, 0, 0))
    side_light2 = PointLight((1, -1, -1), 2, (0, 255, 0))
    top_light = PointLight((0, 0, -2), 2, (255, 255, 255))
    
    sphere = Sphere((0, 0, -1), 0.5, orange_mat)


    # Define Scene
    # scene = Scene(
    #     (
    #         ground,
    #         desk_top1,
    #         desk_top2,
    #         desk_leg1,
    #         desk_leg2,
    #         desk_leg3,
    #         drawer1,
    #         drawer2,
    #         drawer_mid,
    #         drawer_bottom,
    #     ),
    #     (side_light1, side_light2, top_light),
    #     min_distance,
    #     max_distance,
    #     True,
    # )
    
    # eevee = MeshObject((47, 220, -20), 0.05, 'eevee.stl', orange_mat)
    cube = Cube((0, 0, -0.35), 0.7, orange_mat)
    
    scene = Scene(
        (ground, cube),
        (side_light2,),
        min_distance,
        max_distance,
        True
    )
    
    # Create Image
    image = Image.new(mode="RGB", size=image_size)
    render_image = image.load()
    
    # Force Machine code to be compiled
    render(0, 0, scene)

    # Set up Pool
    pool = Pool(processes=4)
    func = partial(renderRow, scene)

    total = []

    # Create Progress Bar
    pbar = tqdm(total=image_height * image_width, unit_scale=True, unit=" pixels")

    for row in pool.imap(func, range(image_height)):
        total.append(row)
        pbar.update(image_width)

    # Close other processes
    pool.close()
    pool.join()

    total = np.array(total) # Convert list to array
    total = total.reshape(image_height, image_width, 3).transpose(1, 0, 2) # Reshape array

    # Save color values to image
    for x in range(image_width):
        for y in range(image_height):
            r = total[x, y, 0]
            g = total[x, y, 1]
            b = total[x, y, 2]
            render_image[x, y] = (int(r), int(g), int(b))

    # Save Render
    image.save("renders/render.png", format="png")

    # Display
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread("renders/render.png")
    plt.imshow(img)
    plt.axis("off")
    plt.show()
