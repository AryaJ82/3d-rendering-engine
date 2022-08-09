import itertools
import pygame

from math import sin, cos


class Vector:
    # coordinates of a 3d vector
    cds = list

    def __init__(self, x: float, y: float, z: float) -> None:
        self.cds = [x, y, z]

    def project(self, mesh_cm: 'Vector', mat_proj: list):
        """ Projects vector onto screen

        <maT_proj> is the projection matrix
        <mesh_cm> is a Vector object
        """
        # actual position of the vector relative to origin instead of cm
        v = mesh_cm + self
        return pm_mult(mat_proj, v)

    def rotate(self, alpha: float, beta: float, theta: float):
        """  Return a new vector rotated the corresponding radians
        around its origin
        """
        x, y, z = self.cds
        a, b, c = x, y, z
        y = b * cos(alpha) + c * sin(alpha)
        z = c * cos(alpha) - b * sin(alpha)

        a, b, c = x, y, z
        x = a * cos(beta) - c * sin(beta)
        z = a * sin(beta) + c * cos(beta)

        a, b, c = x, y, z
        x = a * cos(theta) + b * sin(theta)
        y = b * cos(theta) - a * sin(theta)

        return Vector(x, y, z)

    def normalize(self) -> None:
        """ Normalize *this* vector
        """
        s = sum(i**2 for i in self.cds)
        if s != 0:
            self.cds = list(map(lambda x: x/s, self.cds))

    def dot(self, other: 'Vector') -> float:
        """ Dot product between two vector objects
        """
        return sum(self.cds[i]*other.cds[i] for i in range(3))

    def cross(self, other: 'Vector') -> 'Vector':
        """Cross product of two vectors"""
        v = Vector(self.cds[1] * other.cds[2] - self.cds[2] * other.cds[1],
                   self.cds[2] * other.cds[0] - self.cds[0] * other.cds[2],
                   self.cds[2] * other.cds[1] - self.cds[1] * other.cds[0])
        return v

    def __add__(self, other: 'Vector') -> 'Vector':
        """ Returns a new Vector object; self + other
        <other> is a vector
        """
        v = [0, 0, 0]
        for i in range(3):
            v[i] = self.cds[i] + other.cds[i]
        return Vector(*v)

    def __sub__(self, other: 'Vector') -> 'Vector':
        """ Returns a new Vector object; self - other
        <other> is a vector
        """
        v = [0, 0, 0]
        for i in range(3):
            v[i] = self.cds[i] - other.cds[i]
        return Vector(*v)

    def __repr__(self) -> str:
        return f"""Vector({round(self.cds[0], 2)}, {round(self.cds[1], 2)}, \
        {round(self.cds[2], 2)})"""


class Triangle:
    # list of vectors corresponding to the vertices of the triangle in 3d space
    # defined clockwise relative to the center of mass (normal pointing out)
    vertices: list('Vector')

    # triangle normal
    normal: Vector

    # colour of triangle in RGB
    clr: tuple

    def __init__(self, vertices: list('Vector'), clr: tuple) -> None:
        self.vertices = vertices
        self.normal = Vector.cross(vertices[1] - vertices[0],
                                   vertices[2] - vertices[0])
        self.normal.normalize()
        self.clr = clr

    def draw_prime(self, screen) -> None:
        """ Draws this triangle onto the screen

        Precondition: This triangle has been projected onto the screen
        """
        proj_vert = []
        for v in self.vertices:
            proj_vert.append(tuple(v.cds[:2]))
        pygame.draw.polygon(screen, self.clr, proj_vert)

    def draw(self, screen) -> None:
        """ Draws this triangle onto the screen
        ALTERNATIVE TO Triangle.draw_prime; this one draws a wireframe

        Precondition: This triangle has been projected onto the screen
        """
        proj_vert = []
        for v in self.vertices:
            proj_vert.append(tuple(v.cds[:2]))

        for line in itertools.combinations(proj_vert, 2):
            # print(line)
            pygame.draw.line(screen, (255, 255, 255), *line)

    def project(self, poly_cm: Vector, mat_proj: list) -> 'Triangle':
        """ Returns a new Triangle; this one projected onto the screen
        using the projection matrix
        """
        image = []
        for v in self.vertices:
            image.append(v.project(poly_cm, mat_proj))
        return Triangle(image, self.clr)

    def rotate(self, rot: list) -> None:
        """ Return a new triangle rotated the corresponding radians
        """
        rover = list(map(lambda v: v.rotate(*rot), self.vertices))
        return Triangle(rover, self.clr)

    def __repr__(self):
        return f"[{self.vertices[0]} , {self.vertices[1]}, {self.vertices[2]}]"


class Mesh:
    # Center of mass of mesh. Triangle vertices defined relative to this as
    # origin
    cm: Vector

    # list of Triangles, which are relatively positioned to the center of mass
    # of the Mesh
    triangles: list

    # cumulative rotation value. 3 numbers, represent rotation about the cm
    # in radians
    rotation: list

    def __init__(self, cm, triangles: list) -> None:
        self.cm = cm
        self.triangles = triangles
        self.rotation = [0] * 3

    def draw(self, screen, mat_proj) -> None:
        """ Draw the mesh on the screen
        """
        projTriangles = self._raster(mat_proj)
        # TODO: insufficient logic
        for t in projTriangles:
            t.draw(screen)

    def _raster(self, mat_proj: list):
        """ Return a list of Triangle objects; based on all of the triangles in
        the given(self) mesh that has been projected onto the screen so that
        they can be rendered

        <matProj> is the associated projection matrix
        """
        t = []
        for triangle in self.triangles:
            rot_triangle = triangle.rotate(self.rotation)
            t.append(rot_triangle.project(self.cm, mat_proj))
        return t
        t = []
        # for triangle in self.triangles:
        #     rot_triangle = triangle.rotate(self.rotation)
        #     # only project the triangle if the normal points away from the scrn
        #     if Vector.dot(rot_triangle.normal, (rot_triangle.vertices[0] - Vector(0, 0, 1))) > 0:
        #         t.append(rot_triangle.project(self.cm, mat_proj))
        return t


    def rotate(self, rot: list) -> None:
        """
        Rotate the polygon through its own axis.
        Angles are measured in radians.
        """
        for i in range(3):
            self.rotation[i] += rot[i]
            self.rotation[i] %= 6.28318


def pm_mult(matrix: list, v: Vector) -> Vector:
    """ Multiplies the 4d projection matrix on a 3d vector
    the '4th' element of <v> is presupposed to be 1.
    The first 3 elements of the resultant 4d vector are divided by the 4th
    element and the now 3d vector is returned after some scaling
    """
    ls = [0, 0, 0, 0]
    for i in range(4):
        for k in range(3):
            ls[i] += v.cds[k] * matrix[k][i]
            # print(f"{matrix[k][i]} * {v.cds[k]}")
        ls[i] += matrix[3][i]

    # print(ls)
    if ls[3] != 0:
        for i in range(3):
            ls[i] /= ls[3]
            ls[i] += 1  # changes range from -1 to 1 to 0 to 2
            ls[i] /= 2  # changes scale to 0 to 1
        # TODO: remove hardcoded scaling
        ls[0] *= 600
        ls[1] *= 800
        return Vector(*ls[:3])
    else:
        return Vector(0, 0, 0)


def gen_cube(pos: Vector, s: float) -> Mesh:
    """ Generate regular cube with one edge at <pos> and side length <s>
    """
    # top and bottom of cube
    tb = [Triangle([Vector(0, s, 0), Vector(0, s, s), Vector(s, s, 0)],
                   (125, 125, 125)),
          Triangle([Vector(s, s, 0), Vector(0, s, s), Vector(s, s, s)],
                   (125, 125, 125)),
          Triangle([Vector(s, 0, 0), Vector(0, 0, 0), Vector(0, 0, s)],
                   (255, 40, 125)),
          Triangle([Vector(s, 0, 0), Vector(0, 0, s), Vector(s, 0, s)],
                   (255, 40, 125))]
    # north and south of cube
    ns = [Triangle([Vector(0, s, 0), Vector(s, s, 0), Vector(s, 0, 0)],
                   (0, 255, 0)),
          Triangle([Vector(s, 0, 0), Vector(0, 0, 0), Vector(0, s, 0)],
                   (0, 255, 0)),
          Triangle([Vector(s, 0, s), Vector(s, s, s), Vector(0, s, s)],
                   (0, 0, 255)),
          Triangle([Vector(0, s, s), Vector(0, 0, s), Vector(s, 0, s)],
                   (0, 0, 255))]
    # east and west of cube
    ew = [Triangle([Vector(s, s, s), Vector(s, 0, s), Vector(s, 0, 0)],
                   (255, 255, 255)),
          Triangle([Vector(s, 0, 0), Vector(s, s, 0), Vector(s, s, s)],
                   (255, 255, 255)),
          Triangle([Vector(0, 0, 0), Vector(0, 0, s), Vector(0, s, s)],
                   (255, 0, 0)),
          Triangle([Vector(0, s, s), Vector(0, s, 0), Vector(0, 0, 0)],
                   (255, 0, 0))]
    return Mesh(pos, tb + ns + ew)


def gen_tetra(pos: Vector) -> Mesh:
    """ Generate an irregular tetrahedron with one edge at <pos>
    """
    Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)
    return Mesh(pos, [Triangle()])


def get_square(pos: Vector, s: float) -> Mesh:
    """ Generate a square with center at <pos>, side length of sqrt(2)*s
    """
    return Mesh(pos, [
        Triangle([Vector(s, 0, 0), Vector(0, s, 0), Vector(-s, 0, 0)],
                 (255, 255, 255)),
        Triangle([Vector(-s, 0, 0), Vector(0, -s, 0), Vector(s, 0, 0)],
                 (255, 255, 255))])