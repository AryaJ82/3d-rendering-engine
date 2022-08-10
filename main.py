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


def redraw(screen):
    screen.fill((0, 0, 0))

    cube.draw(screen, mat_proj)
    # pygame.draw.line(screen, (255, 255, 255), *line, width=5)
    pygame.display.flip()


cube = gen_cube(Vector(0, 0, 30), 10)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        cube.rotate([0.001, 0.002, 0.003])
    redraw(screen)

pygame.quit()
exit()
