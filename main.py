from mesh import *
from sys import exit
from math import tan

pygame.init()

# constants
W = 800  # width
H = 800  # height
ASP = W / H  # aspect ratio
Near = 0.1  # closest 'player' can see
Far = 1000  # farthest 'player' can see
Fov = 3.14159 / 2  # field of view, 90 degrees
FovRad = (tan(Fov / 2)) ** -1  # calculation for convenience

# Projection Matrix (4x4)
# Written: [row][column] top to bottom, left to right
mat_proj = [[0.0] * 4 for i in range(4)]
mat_proj[0][0] = ASP * FovRad
mat_proj[1][1] = FovRad
mat_proj[2][2] = Far / (Far - Near)
mat_proj[3][2] = (-Far * Near) / (Far - Near)
mat_proj[2][3] = 1

screen = pygame.display.set_mode((W, H))

## Camera vectors
# vector representing camera position; at origin by default
camera_pos = Vector(0, 0, 0)
# direction camera is pointing; along z axis by default
camera_dir = Vector(0, 0, 1)
# up
up = Vector(0, 1, 0)
right = Vector.cross(up, camera_dir)
target = camera_pos + camera_dir

mat_camera = point_at_matrix(camera_pos, target, up)
mat_view = quick_invert(mat_camera)

def redraw(screen):
    screen.fill((0, 0, 0))

    cube.draw(screen, mat_proj, mat_view)
    pygame.display.flip()

## TODO: function update_camera

cube = gen_cube(Vector(0, 0, 40), 10)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        cube.rotate([0.002, 0.003, 0.004])
    elif keys[pygame.K_w]:
        # vertical elevation
        camera_pos -= up.sc_mult(0.1)
    elif keys[pygame.K_s]:
        # vertical depression
        camera_pos += up.sc_mult(0.1)
    elif keys[pygame.K_z]:
        # horizontal yaw
        camera_dir = camera_dir.rotate(0.0, 0.005, 0.0)
    elif keys[pygame.K_x]:
        # horizontal yaw
        camera_dir = camera_dir.rotate(0.0, -0.005, 0.0)
    elif keys[pygame.K_UP]:
        # camera position movement
        camera_pos += camera_dir.sc_mult(0.1)
    elif keys[pygame.K_DOWN]:
        # camera position movement
        camera_pos -= camera_dir.sc_mult(0.1)
    elif keys[pygame.K_RIGHT]:
        # camera position movement
        camera_pos += right.sc_mult(0.1)
    elif keys[pygame.K_LEFT]:
        # camera position movement
        camera_pos -= right.sc_mult(0.1)

    # TODO: only redefine right when necessary
    right = Vector.cross(up, camera_dir)
    right.normalize()

    target = camera_pos + camera_dir
    mat_camera = point_at_matrix(camera_pos, target, up)
    mat_view = quick_invert(mat_camera)

    redraw(screen)

pygame.quit()
exit()
