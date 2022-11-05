import time

from mesh import *
from sys import exit
from math import tan

pygame.init()

# ~~ Constants
W = 800  # width of screen
H = 800  # height of screen
Near = 0.1  # closest 'player' can see
Far = 1000  # farthest 'player' can see
Fov = 3.14159 / 2  # field of view, 90 degrees

# ~~ Create projection Matrix (4x4)
# Written: [row][column] top to bottom, left to right
mat_proj = [[0.0] * 4 for _ in range(4)]
mat_proj[0][0] = (W / H) * ((tan(Fov / 2)) ** -1)
mat_proj[1][1] = ((tan(Fov / 2)) ** -1)
mat_proj[2][2] = Far / (Far - Near)
mat_proj[2][3] = (-Far * Near) / (Far - Near)
mat_proj[3][2] = 1

# ~~ Camera vectors
# camera position; at origin by default
camera_pos = [0, 0, 0]

# direction camera is pointing; along z axis by default
camera_dir = [0, 0, 1]

# up direction; along y axis by default
up = [0, 1, 0]

# right direction; ortho to up and camera_dir
right = vCross(up, camera_dir)

# ~~ View change matrix
# mat_camera = point_at_matrix(camera_pos, camera_pos + camera_dir, up, right)
# mat_view = quick_invert(mat_camera)
mat_view = get_viewmat(camera_pos, camera_dir, up, right)

# Create screen
screen = pygame.display.set_mode((W, H))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def redraw(scrn):
    screen.fill((0, 0, 0))

    for o in objects:
        o.draw(scrn, mat_proj, mat_view)

    pygame.display.flip()


def camera_movement(keys) -> List[List[float]]:
    """ Handle the movement for the camera. Returns a new view transformation
    matrix.
    """
    global camera_pos
    global camera_dir
    global right

    if keys[pygame.K_SPACE]:
        for o in objects:
            o.rotate([0.002, 0.003, 0.004])

    # Separating <if> statements into negative/positive movement directions
    # allows for movement in multiple directions at once

    # vertical elevation/depression
    if keys[pygame.K_w]:
        camera_pos = vSub(camera_pos, vScMult(up, 0.1))
    elif keys[pygame.K_s]:
        camera_pos = vAdd(camera_pos, vScMult(up, 0.1))

    # horizontal yaw
    if keys[pygame.K_z]:
        camera_dir = rad_rotate(camera_dir, 0.0, 0.01, 0.0)
        # Right direction only changes with yaw
        right = vCross(up, camera_dir)
        vNormalize(right)
    elif keys[pygame.K_x]:
        camera_dir = rad_rotate(camera_dir, 0.0, -0.01, 0.0)
        # Right direction only changes with yaw
        right = vCross(up, camera_dir)
        vNormalize(right)

    # camera position (forward and backward) movement
    if keys[pygame.K_UP]:
        camera_pos = vAdd(camera_pos, vScMult(camera_dir, 0.1))
    elif keys[pygame.K_DOWN]:
        camera_pos = vSub(camera_pos, vScMult(camera_dir, 0.1))

    # camera position (right and left) movement
    if keys[pygame.K_RIGHT]:
        camera_pos = vAdd(camera_pos, vScMult(right, 0.1))
    elif keys[pygame.K_LEFT]:
        # camera position movement
        camera_pos = vSub(camera_pos, vScMult(right, 0.1))

    return get_viewmat(camera_pos, camera_dir, up, right)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

objects = []
objects.append(file_to_mesh([0, 2, 7], r".\Assets\teapot.obj"))
objects[0].rotate([0, 0, 3.1415])

start = time.time()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    mat_view = camera_movement(pygame.key.get_pressed())

    redraw(screen)
    end = time.time()
    print(f"Tick time: {end - start}")
    start = end

pygame.quit()
exit()
