import itertools
import pygame

from math import sin, cos
from typing import List
from bisect import insort


class Vector:
    # coordinates of a 3d vector
    cds = list

    def __init__(self, x: float, y: float, z: float) -> None:
        self.cds = [x, y, z]

    def project(self, mesh_ro: 'Vector',
                mat_proj: List[List[float]]) -> 'Vector':
        """ Projects vector onto screen

        <mat_proj> is the projection matrix
        <mesh_ro> is a Vector; the relative origin of the parent Mesh

        Vector should have been translated into view space
        """
        # actual position of the vector relative to origin instead of ro
        v = mmult(mat_proj, mesh_ro + self)
        z = mesh_ro.cds[2] + self.cds[2]
        if z != 0:
            for i in range(3):
                v.cds[i] /= z
                v.cds[i] += 1  # changes range from -1 to 1 to 0 to 2
                v.cds[i] /= 2  # changes scale to 0 to 1
            # TODO: remove hardcoded scaling
            v.cds[0] *= 800
            v.cds[1] *= 800
        return v

    def rotate(self, rot: List[List[float]]) -> 'Vector':
        """  Return a new vector rotated the corresponding radians
        around its origin
        <rot> is a list of list of cos and sin calculations done on the
        angles the vector is to be rotated

        Specifically:
        rot = [[cos(self.rotation[0]), sin(self.rotation[0])],
               [cos(self.rotation[1]), sin(self.rotation[1])],
               [cos(self.rotation[2]), sin(self.rotation[2])]]
        """
        x, y, z = self.cds
        a, b, c = x, y, z
        # y = b * cos(alpha) + c * sin(alpha)
        # z = c * cos(alpha) - b * sin(alpha)
        y = b * rot[0][0] + c * rot[0][1]
        z = c * rot[0][0] - b * rot[0][1]

        a, b, c = x, y, z
        # x = a * cos(beta) - c * sin(beta)
        # z = a * sin(beta) + c * cos(beta)
        x = a * rot[1][0] - c * rot[1][1]
        z = a * rot[1][1] + c * rot[1][0]

        a, b, c = x, y, z
        # x = a * cos(theta) + b * sin(theta)
        # y = b * cos(theta) - a * sin(theta)
        x = a * rot[2][0] + b * rot[2][1]
        y = b * rot[2][0] - a * rot[2][1]

        return Vector(x, y, z)

    def rad_rotate(self, alpha: float, beta: float, theta: float) -> 'Vector':
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
        """ Normalize *this* vector"""
        s = sum(i ** 2 for i in self.cds) ** 0.5
        if s != 0:
            self.cds = list(map(lambda x: x / s, self.cds))

    def dot(self, other: 'Vector') -> float:
        """ Dot product between two vector objects"""
        return sum(self.cds[i] * other.cds[i] for i in range(3))

    def cross(self, other: 'Vector') -> 'Vector':
        """Cross product of two vectors"""
        v = Vector(self.cds[1] * other.cds[2] - self.cds[2] * other.cds[1],
                   self.cds[2] * other.cds[0] - self.cds[0] * other.cds[2],
                   self.cds[0] * other.cds[1] - self.cds[1] * other.cds[0])
        return v

    def sc_mult(self, p: float) -> 'Vector':
        """ New scalar multiplication of vector by <p>"""
        return Vector(self.cds[0] * p, self.cds[1] * p, self.cds[2] * p)

    def __add__(self, other: 'Vector') -> 'Vector':
        """ Vector addition """
        v = [0, 0, 0]
        for i in range(3):
            v[i] = self.cds[i] + other.cds[i]
        return Vector(*v)

    def __sub__(self, other: 'Vector') -> 'Vector':
        """ Vector subtraction; self - other"""
        v = [0, 0, 0]
        for i in range(3):
            v[i] = self.cds[i] - other.cds[i]
        return Vector(*v)

    def __eq__(self, other: 'Vector') -> bool:
        return (round(self.cds[0] - other.cds[0], 3) == 0.000) and (
                round(self.cds[1] - other.cds[1], 3) == 0.000) and (
                       round(self.cds[2] - other.cds[2], 3) == 0.000)

    def __repr__(self) -> str:
        return (f"Vector({round(self.cds[0], 2)}, "
                f"{round(self.cds[1], 2)}, "
                f"{round(self.cds[2], 2)})")


class Triangle:
    # list of vectors corresponding to the vertices of the triangle in 3d space
    # defined clockwise relative to the relative origin (normal pointing out)
    vertices: List['Vector']

    # triangle normal; over written to the parent's normal if the triangle has
    # been projected. Generated as needed.
    normal: Vector

    # triangle center of mass; over written to the parent's if the triangle
    # has been projected. Generated as needed. Relative to origin of vertices,
    # not to absolute (world) origin.
    cm: Vector

    # colour of triangle in RGB
    clr: tuple

    def __init__(self, vertices: List['Vector'], clr: tuple) -> None:
        self.vertices = vertices
        self.clr = clr

    def gen_normal(self):
        """ Sets the normal for this triangle """
        self.normal = Vector.cross(self.vertices[1] - self.vertices[0],
                                   self.vertices[2] - self.vertices[0])

    def gen_cm(self):
        """ Find the center of mass of this triangle"""
        self.cm = (self.vertices[0] + self.vertices[1] + self.vertices[
            2]).sc_mult(0.3333)

    def draw(self, screen) -> None:
        """ Draws this triangle onto the screen

        Precondition: This triangle has already been projected onto the screen
        """
        # find vertex positions on screen
        proj_vert = []
        for v in self.vertices:
            proj_vert.append(tuple(v.cds[:2]))

        # shading, light comes from camera
        light_dir = Vector(0, 0, -1)
        clr = sc_mult(self.clr, abs(self.normal.dot(light_dir)))

        # draw triangle
        pygame.draw.polygon(screen, clr, proj_vert)

    def draw_prime(self, screen) -> None:
        """ Draws this triangle onto the screen
        ALTERNATIVE TO Triangle.draw; this one draws a wireframe

        Precondition: This triangle has been projected onto the screen
        """
        proj_vert = []
        for v in self.vertices:
            proj_vert.append(tuple(v.cds[:2]))

        for line in itertools.combinations(proj_vert, 2):
            pygame.draw.line(screen, (255, 255, 255), *line)

    def project(self, mesh_ro: Vector,
                mat_proj: List[List[float]]) -> 'Triangle':
        """ Returns a new Triangle; this one projected onto the screen
        using the projection matrix

        Precondition:
        The normal for this triangle has been generated
        The cm of this triangle as been generated
        The triangle has been transformed into view space
        """
        image = []
        for v in self.vertices:
            image.append(v.project(mesh_ro, mat_proj))
        t = Triangle(image, self.clr)

        # normal used in shading, projected triangle must have the same normal
        # as it's parent triangle to do this correctly
        t.normal = self.normal

        # cm of projected triangle set to this one's necessary for painters alg
        t.cm = self.cm
        return t

    def view_transform(self, mat_view: List[List[float]]) -> 'Triangle':
        """ Returns a new triangle as it would be if the current's vertices
        were transformed into view space using mat_view
        """
        vertices = []
        for v in self.vertices:
            vertices.append(mmult(mat_view, v))
        return Triangle(vertices, self.clr)

    def rotate(self, rot: List[List[float]]) -> 'Triangle':
        """ Return a new triangle rotated the corresponding radians
        """
        rover = list(map(lambda v: v.rotate(rot), self.vertices))
        return Triangle(rover, self.clr)

    def __repr__(self):
        return f"[{self.vertices[0]} , {self.vertices[1]}, {self.vertices[2]}]"


class Mesh:
    # Relative origin of mesh. Triangle vertices defined relative to this as
    # origin
    ro: Vector

    # list of Triangles, which are relatively positioned to relative origin (ro)
    # of the Mesh
    # All triangles in self.triangles must have their normals generated
    triangles: List[Triangle]

    # cumulative rotation value. 3 numbers, represent rotation about the ro
    # in radians (between -2pi to 2pi)
    rotation: List[float]

    def __init__(self, ro, triangles: List[Triangle]) -> None:
        self.ro = ro
        self.triangles = triangles
        self.rotation = [0] * 3

    def draw(self, screen, mat_proj: List[List[float]],
             mat_view: List[List[float]]) -> None:
        """ Draw the mesh on the screen

        <mat_proj> is the associated projection matrix
        <mat_view> the view transform matrix
        """
        proj_triangles = self._raster(mat_proj, mat_view)

        for t in proj_triangles:
            t.draw(screen)

    def _raster(self, mat_proj: List[List[float]],
                mat_view: List[List[float]]) -> List[Triangle]:
        """ Return a list of Triangle objects; based on all of the triangles in
        the given(self) mesh that has been projected onto the screen so that
        they can be rendered

        <mat_proj> is the associated projection matrix
        <mat_view> the view transform matrix
        """
        t = []
        # transform relative origin into view space
        view_ro = mmult(mat_view, self.ro)

        # trig operations for vector rotation, done in advance as op is slow
        rot = [[cos(self.rotation[0]), sin(self.rotation[0])],
               [cos(self.rotation[1]), sin(self.rotation[1])],
               [cos(self.rotation[2]), sin(self.rotation[2])]]

        for triangle in self.triangles:
            # rotate and transform the normal of this triangle
            view_norm = norm_mmult(mat_view, triangle.normal.rotate(rot))

            # TODO: is the below norm_mmult or mmult?
            # process triangle only if its normal points away from the screen
            if view_norm.dot(view_ro + mmult(mat_view,
                                             triangle.vertices[0].rotate(
                                                 rot))) < 0:
                # rotate and transform triangle
                view_tri = triangle.rotate(rot).view_transform(mat_view)

                # set the triangles normal and center of mass
                view_tri.normal = view_norm
                view_tri.gen_cm()

                insort(t, view_tri.project(view_ro, mat_proj),
                       key=lambda x: -x.cm.cds[2])
        return t

    def rotate(self, rot: List[float]) -> None:
        """
        Rotate the polygon through its own axes.
        Angles are measured in radians.
        """
        for i in range(3):
            self.rotation[i] += rot[i]
            self.rotation[i] %= 6.28318


def sc_mult(tup: tuple, sc: float) -> tuple:
    """ Multiplies the tuples entries by <sc>, rounds to int, and returns"""
    return tuple(map(lambda x: int(x * sc), tup))


def mmult(matrix: List[List[float]], v: Vector) -> Vector:
    """ Multiplies the 4x4 <matrix> with a 3d vector <v>. The '4th' element of
    <v> is presupposed to be 1. Return the resulting 3d vector
    """
    # ls = [0.0] * 3
    # for i in range(3):
    #     for k in range(3):
    #         ls[i] += v.cds[k] * matrix[k][i]
    #     ls[i] += matrix[3][i]
    #
    # return ls

    # Code performs the same function as above, but about 0.03 seconds faster
    ls = [0.0] * 3
    ls[0] = v.cds[0] * matrix[0][0] + v.cds[1] * matrix[1][0] + v.cds[2] * \
            matrix[2][0] + matrix[3][0]
    ls[1] = v.cds[0] * matrix[0][1] + v.cds[1] * matrix[1][1] + v.cds[2] * \
            matrix[2][1] + matrix[3][1]
    ls[2] = v.cds[0] * matrix[0][2] + v.cds[1] * matrix[1][2] + v.cds[2] * \
            matrix[2][2] + matrix[3][2]

    return Vector(*ls)


def norm_mmult(matrix: List[List[float]], v: Vector) -> Vector:
    """ Multiplies the 4x4 <matrix> with a 3d vector <v>. The '4th' element of
    <v> is presupposed to be 0. Return the resulting 3d vector
    Do not mix up between this function and mmult().
    """
    # Code performs the same function as above, but about 0.03 seconds faster
    ls = [0.0] * 3
    ls[0] = v.cds[0] * matrix[0][0] + v.cds[1] * matrix[1][0] + v.cds[2] * \
            matrix[2][0]
    ls[1] = v.cds[0] * matrix[0][1] + v.cds[1] * matrix[1][1] + v.cds[2] * \
            matrix[2][1]
    ls[2] = v.cds[0] * matrix[0][2] + v.cds[1] * matrix[1][2] + v.cds[2] * \
            matrix[2][2]

    return Vector(*ls)


def get_viewmat(pos: Vector, forward: Vector, up: Vector, right: Vector) \
        -> List[List[float]]:
    """ Returns the view matrix. The view matrix is a change of basis matrix
    from the standard basis to view basis. It cannot strictly be called a
    linear transformation as it also changes the origin.

    Origin changes to <pos>
    (0, 0, 1) maps to <dir>
    y-axis morally mapped to <up>
    x-axis mapped to x <right>
    """
    # keep the up direction orthonormal to <dir> and <right>
    new_up = up - forward.sc_mult(Vector.dot(forward, up))
    new_up.normalize()

    matrix = [[0.0] * 4 for _ in range(4)]
    matrix[0][0], matrix[1][0], matrix[2][0] = right.cds

    matrix[0][1], matrix[1][1], matrix[2][1] = new_up.cds

    matrix[0][2], matrix[1][2], matrix[2][2] = forward.cds

    matrix[3][0] = - pos.dot(right)
    matrix[3][1] = - pos.dot(new_up)
    matrix[3][2] = - pos.dot(forward)

    matrix[3][3] = 1.0

    return matrix


def gen_cube(pos: Vector, s: float) -> Mesh:
    """ Generate regular cube with one edge at <pos> and side length <s>
    """
    # top and bottom of cube
    tb = [Triangle([Vector(0, s, 0), Vector(0, s, s), Vector(s, s, s)],
                   (125, 125, 125)),
          Triangle([Vector(0, s, 0), Vector(s, s, s), Vector(s, s, 0)],
                   (125, 125, 125)),
          Triangle([Vector(s, 0, s), Vector(0, 0, s), Vector(0, 0, 0)],
                   (255, 40, 125)),
          Triangle([Vector(s, 0, s), Vector(0, 0, 0), Vector(s, 0, 0)],
                   (255, 40, 125))]
    # north and south of cube
    ns = [Triangle([Vector(0, 0, 0), Vector(0, s, 0), Vector(s, s, 0)],
                   (0, 255, 0)),
          Triangle([Vector(0, 0, 0), Vector(s, s, 0), Vector(s, 0, 0)],
                   (0, 255, 0)),
          Triangle([Vector(s, 0, s), Vector(s, s, s), Vector(0, s, s)],
                   (0, 0, 255)),
          Triangle([Vector(s, 0, s), Vector(0, s, s), Vector(0, 0, s)],
                   (0, 0, 255))]
    # east and west of cube
    ew = [Triangle([Vector(s, 0, 0), Vector(s, s, 0), Vector(s, s, s)],
                   (255, 255, 255)),
          Triangle([Vector(s, 0, 0), Vector(s, s, s), Vector(s, 0, s)],
                   (255, 255, 255)),
          Triangle([Vector(0, 0, s), Vector(0, s, s), Vector(0, s, 0)],
                   (255, 0, 0)),
          Triangle([Vector(0, 0, s), Vector(0, s, 0), Vector(0, 0, 0)],
                   (255, 0, 0))]
    return Mesh(pos, tb + ns + ew)


def file_to_mesh(pos: Vector, d: str):
    """ Generate a mesh from the .obj file at <d> located at <pos>"""
    vertices = []
    triangles = []
    with open(d, 'r') as f:
        # vertex entries take the form "v float float float"
        line = f.readline().split(" ")
        while line[0] == "v":
            vertices.append(Vector(*list(map(float, line[1:]))))
            line = f.readline().split(" ")

        # blank line
        line = f.readline().split(" ")

        # triangle entries take the form "t int int int"
        # where int correspond to an entry in <vertices>
        while line[0] == "f":
            t = Triangle(list(map(lambda n: vertices[int(n) - 1], line[1:])),
                         (255, 255, 255))
            t.gen_normal()
            t.normal.normalize()
            triangles.append(t)
            line = f.readline().split(" ")

    return Mesh(pos, triangles)
