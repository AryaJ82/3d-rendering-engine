from mesh import *
from sys import exit
from math import tan

pygame.init()

# constants
W = 800  # width
H = 800  # height
Near = 0.1  # closest 'player' can see
Far = 1000  # farthest 'player' can see
Fov = 3.14159 / 2  # field of view, 90 degrees

# Create projection Matrix (4x4)
# Written: [row][column] top to bottom, left to right
mat_proj = [[0.0] * 4 for i in range(4)]
mat_proj[0][0] = (W / H) * ((tan(Fov / 2)) ** -1)
mat_proj[1][1] = ((tan(Fov / 2)) ** -1)
mat_proj[2][2] = Far / (Far - Near)
mat_proj[3][2] = (-Far * Near) / (Far - Near)
mat_proj[2][3] = 1

# Camera vectors
# camera position; at origin by default
camera_pos = Vector(0, 0, 0)

# direction camera is pointing; along z axis by default
camera_dir = Vector(0, 0, 1)
# up direction; along y axis by default
up = Vector(0, 1, 0)
# right direction
right = Vector.cross(up, camera_dir)
# up, right and camera_dir form a basis and are used for transformations

mat_camera = point_at_matrix(camera_pos, camera_pos + camera_dir, up, right)
mat_view = quick_invert(mat_camera)

# Create screen
screen = pygame.display.set_mode((W, H))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def redraw(screen):
    screen.fill((0, 0, 0))

    cube.draw(screen, mat_proj, mat_view)
    pygame.display.flip()


def camera_movement(keys: dict) -> List[List[float]]:
    """ Handle the movement for the camera. Returns a new view transformation
    matrix.
    """
    global camera_pos
    global camera_dir
    global right

    if keys[pygame.K_SPACE]:
        cube.rotate([0.002, 0.003, 0.004])

    # Separating <if> statements into negative/positive movement directions
    # allows for movement in multiple directions at once

    # vertical elevation/depression
    if keys[pygame.K_w]:
        camera_pos -= up.sc_mult(0.1)
    elif keys[pygame.K_s]:
        camera_pos += up.sc_mult(0.1)

    # horizontal yaw
    if keys[pygame.K_z]:
        camera_dir = camera_dir.rotate(0.0, 0.005, 0.0)
        # Right direction only changes with differing yaw
        right = Vector.cross(up, camera_dir)
        right.normalize()
    elif keys[pygame.K_x]:
        camera_dir = camera_dir.rotate(0.0, -0.005, 0.0)
        # Right direction only changes with differing yaw
        right = Vector.cross(up, camera_dir)
        right.normalize()

    # camera position (forward and backward) movement
    if keys[pygame.K_UP]:
        camera_pos += camera_dir.sc_mult(0.1)
    elif keys[pygame.K_DOWN]:
        camera_pos -= camera_dir.sc_mult(0.1)

    # camera position (right and left) movement
    if keys[pygame.K_RIGHT]:
        camera_pos += right.sc_mult(0.1)
    elif keys[pygame.K_LEFT]:
        # camera position movement
        camera_pos -= right.sc_mult(0.1)

    return quick_invert(
        point_at_matrix(camera_pos, camera_pos + camera_dir, up, right))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cube = gen_cube(Vector(0, 0, 40), 10)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    mat_view = camera_movement(pygame.key.get_pressed())

    redraw(screen)

pygame.quit()
exit()
