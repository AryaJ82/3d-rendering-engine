import itertools
import pygame

from math import sin, cos
from typing import List
from cuda_code import *
from bisect import insort
import numpy


def project(vector: List[float], mesh_ro, mat_proj: List[List[float]]):
    """ Projects vector onto screen

    <mat_proj> is the projection matrix
    <mesh_ro> is a Vector; the relative origin of the parent Mesh

    Vector should have been translated into view space
    """
    # actual position of the vector relative to origin instead of ro
    v = mmult(mat_proj, vAdd(vector, mesh_ro))
    z = mesh_ro[2] + vector[2]  # scaled by z for parallax
    if z != 0:
        for i in range(3):
            v[i] /= z
            v[i] += 1  # changes range from -1 to 1 to 0 to 2
            v[i] /= 2  # changes scale to 0 to 1
        # TODO: remove hardcoded scaling
        v[0] *= 800
        v[1] *= 800
    return v


def vRotate(vector: List[float], rot: List[List[float]]) -> List[float]:
    """  Return a new vector rotated the corresponding radians
    around its origin
    <rot> is a list of list of cos and sin calculations done on the
    angles the vector is to be rotated

    Specifically:
    rot = [[cos(self.rotation[0]), sin(self.rotation[0])],
           [cos(self.rotation[1]), sin(self.rotation[1])],
           [cos(self.rotation[2]), sin(self.rotation[2])]]
    """
    x, y, z = vector
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

    # return numpy.array([x, y, z], dtype=numpy.float32)
    return [x, y, z]


def rad_rotate(vector: List[float], alpha: float, beta: float, theta: float) -> \
List[float]:
    """  Return a new vector rotated the corresponding radians
    around its origin
    """
    x, y, z = vector
    a, b, c = x, y, z
    y = b * cos(alpha) + c * sin(alpha)
    z = c * cos(alpha) - b * sin(alpha)

    a, b, c = x, y, z
    x = a * cos(beta) - c * sin(beta)
    z = a * sin(beta) + c * cos(beta)

    a, b, c = x, y, z
    x = a * cos(theta) + b * sin(theta)
    y = b * cos(theta) - a * sin(theta)

    return [x, y, z]


def vNormalize(vector: List[float]) -> None:
    """ Normalize *this* vector"""
    s = sum(i ** 2 for i in vector) ** 0.5
    if s != 0:
        vector[0] /= s
        vector[1] /= s
        vector[2] /= s


def vDot(v: List[float], w: List[float]) -> float:
    """ Dot product between two vector objects"""
    return sum(v[i] * w[i] for i in range(3))


def vCross(v: List[float], w: List[float]) -> List[float]:
    """Cross product of two vectors"""
    v = [v[1] * w[2] - v[2] * w[1],
         v[2] * w[0] - v[0] * w[2],
         v[0] * w[1] - v[1] * w[0]]
    return v


def vScMult(v: List[float], p: float) -> List[float]:
    """ New scalar multiplication of vector by <p>"""
    return [v[0] * p, v[1] * p, v[2] * p]


def vAdd(v: List[float], w: List[float]):
    """ Vector addition """
    p = [0.0, 0.0, 0.0]
    for i in range(3):
        p[i] = v[i] + w[i]
    return p


def vSub(v: List[float], w: List[float]) -> List[float]:
    """ Vector subtraction; v - w"""
    p = [0, 0, 0]
    for i in range(3):
        p[i] = v[i] - w[i]
    return p


def vEq(v: List[float], w: List[float]) -> bool:
    return (round(v[0] - w[0], 3) == 0.000) and (
            round(v[1] - w[1], 3) == 0.000) and (
                   round(v[2] - w[2], 3) == 0.000)


def gen_normal(triangle: List[List[float]]):
    """ Sets the normal for this triangle """
    triangle[0] = vCross(vSub(triangle[2], triangle[1]),
                         vSub(triangle[3], triangle[1]))


def triangle_draw(triangle: List[List[float]], screen) -> None:
    """ Draws this triangle onto the screen

    Precondition: This triangle has already been projected onto the screen
    """
    # find vertex positions on screen
    proj_vert = []
    for i in range(1, 4):
        proj_vert.append(tuple(triangle[i][:2]))

    # shading, light comes from camera direction
    clr = tScMult((255, 255, 255), abs(vDot(triangle[0], [0, 0, -1])))

    # draw triangle
    pygame.draw.polygon(screen, clr, proj_vert)


def triangle_project(triangle: List[List[float]], mesh_ro: List[float],
            mat_proj: List[List[float]]) -> List[List[float]]:
    """ Returns a new Triangle; this one projected onto the screen
    using the projection matrix

    Precondition:
    The triangle has been transformed into view space
    """
    # normal is used in shading, projected triangle must have the same normal
    # as it's parent triangle to do this correctly
    imageTriangle = [triangle[0]]
    for i in range(1, 4):
        imageTriangle.append(project(triangle[i], mesh_ro, mat_proj))

    # TODO: center of mass handling
    # # cm of projected triangle set to this one's necessary for painters alg
    # t.cm = self.cm
    return imageTriangle


def view_transform(triangle: List[List[float]], mat_view: List[List[float]]) -> \
List[List[float]]:
    """ Returns a new triangle as it would be if the current's vertices
    were transformed into view space using mat_view
    """
    # norm_mmult only changes the orientation of the vector, does not modify
    # it's origin
    viewTriangle = [norm_mmult(mat_view, triangle[0])]
    for i in range(1, 4):
        viewTriangle.append(mmult(mat_view, triangle[i]))
    return viewTriangle


def triangle_rotate(triangle:List[List[float]], rot: List[List[float]]) -> List[List[float]]:
    """ Return a new triangle rotated the corresponding radians
    """
    return list(map(lambda v: vRotate(v, rot), triangle))


class Mesh:
    # Relative origin of mesh. Triangle vertices defined relative to this as
    # origin
    ro: List[float]

    # list of Triangles, which are relatively positioned to relative origin (ro)
    # of the Mesh
    # All triangles in self.triangles must have their normals generated
    triangles: List[List[List[float]]]

    # cumulative rotation value. 3 numbers, represent rotation about the ro
    # in radians (between -2pi to 2pi)
    rotation: List[float]

    def __init__(self, ro: List[float], triangles: List[List[List[float]]]):
        self.ro = ro
        self.triangles = triangles
        self.rotation = [0] * 3

    def draw(self, screen, mat_proj: List[List[float]],
             mat_view: List[List[float]]) -> None:
        """ Draw the mesh on the screen

        <mat_proj> is the associated projection matrix
        <mat_view> the view transform matrix
        """
        # proj_triangles = self._raster(mat_proj, mat_view)
        view_ro = mmult(mat_view, self.ro)
        rot = [[cos(self.rotation[0]), sin(self.rotation[0])],
               [cos(self.rotation[1]), sin(self.rotation[1])],
               [cos(self.rotation[2]), sin(self.rotation[2])]]
        ta = numpy.array(self.triangles, dtype=numpy.float32)
        proj_triangles = CUDA_raster(ta, view_ro, rot, mat_view, mat_proj)

        # further triangle processing
        for t in proj_triangles:
            triangle_draw(t, screen)

    def _raster(self, mat_proj: List[List[float]],
                mat_view: List[List[float]]) -> List[List[List[float]]]:
        """ Return a list of Triangle objects; based on all of the triangles in
        the given(self) mesh that has been projected onto the screen so that
        they can be rendered

        <mat_proj> is the associated projection matrix
        <mat_view> the view transform matrix
        """
        # draw queue of triangles
        t = []
        # transform relative origin into view space
        view_ro = mmult(mat_view, self.ro)

        # trig operations for vector rotation, done in advance as op is slow
        rot = [[cos(self.rotation[0]), sin(self.rotation[0])],
               [cos(self.rotation[1]), sin(self.rotation[1])],
               [cos(self.rotation[2]), sin(self.rotation[2])]]

        for triangle in self.triangles:
            # rotate and transform the normal of this triangle
            view_norm = norm_mmult(mat_view, vRotate(triangle[0], rot))

            # process triangle only if its normal points away from the screen
            if vDot(view_norm, vAdd(view_ro, mmult(mat_view, vRotate(triangle[1], rot)))) < 0:
                # rotate and view transform triangle
                view_tri = view_transform(triangle_rotate(triangle, rot), mat_view)
                # view_tri = triangle.rotate(rot).view_transform(mat_view)
                # TODO: painter's algorithm
                # set the triangles normal and center of mass
                # view_tri.normal = view_norm
                # view_tri.gen_cm()

                # project triangle and add it to the draw queue
                # See Triangle.__lt__ for relevant operator overloading for the
                # bisect.insort method
                #insort(t, view_tri.project(view_ro, mat_proj))
                t.append(triangle_project(view_tri, view_ro, mat_proj))
        t.sort(key=lambda x: x[1][2] + x[2][2]+x[3][2], reverse=True)
        return t

    def rotate(self, rot: List[float]) -> None:
        """
        Rotate the polygon through its own axes.
        Angles are measured in radians.
        """
        for i in range(3):
            self.rotation[i] += rot[i]
            self.rotation[i] %= 6.28318


def tScMult(tup: tuple, sc: float) -> tuple:
    """ Multiplies the tuples entries by <sc>, rounds to int, and returns"""
    return tuple(map(lambda x: int(x * sc), tup))


def mmult(matrix: List[List[float]], v: List[float]) -> List[float]:
    """ Multiplies the 4x4 <matrix> with a 3d vector <v>. The '4th' element of
    <v> is presupposed to be 1. Return the resulting 3d vector.
    Functions with a 4x3 vector
    """
    ls = [0.0] * 3
    for i in range(3):
        for k in range(3):
            ls[i] += v[k] * matrix[i][k]
        ls[i] += matrix[i][3]

    return ls


def norm_mmult(matrix: List[List[float]], v: List[float]) -> List[float]:
    """ Multiplies the 4x4 <matrix> with a 3d vector <v>. The '4th' element of
    <v> is presupposed to be 0. Return the resulting 3d vector
    Do not mix up between this function and mmult().
    """

    ls = [0.0] * 3
    for i in range(3):
        for k in range(3):
            ls[i] += v[k] * matrix[i][k]

    return ls


def get_viewmat(pos: List[float], forward: List[float], up: List[float],
                right: List[float]) \
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
    new_up = vSub(up, vScMult(forward, vDot(forward, up)))
    vNormalize(new_up)

    matrix = [[0.0] * 4 for _ in range(4)]
    matrix[0][0], matrix[0][1], matrix[0][2] = right

    matrix[1][0], matrix[1][1], matrix[1][2] = new_up

    matrix[2][0], matrix[2][1], matrix[2][2] = forward

    matrix[0][3] = -vDot(pos, right)
    matrix[1][3] = -vDot(pos, new_up)
    matrix[2][3] = -vDot(pos, forward)

    matrix[3][3] = 1.0

    return matrix


def file_to_mesh(pos: List[float], d: str):
    """ Generate a mesh from the .obj file at <d> located at <pos>"""
    vertices = []
    triangles = []
    with open(d, 'r') as f:
        # vertex entries take the form "v float float float"
        line = f.readline().split(" ")
        while line[0] == "v":
            vertices.append(list((map(float, line[1:]))))
            line = f.readline().split(" ")

        # blank line
        line = f.readline().split(" ")

        # triangle entries take the form "t int int int"
        # where int correspond to an entry in <vertices>
        while line[0] == "f":
            t = [[]] + list(map(lambda n: vertices[int(n) - 1], line[1:]))
            gen_normal(t)
            vNormalize(t[0])
            triangles.append(t)
            line = f.readline().split(" ")

    return Mesh(pos, triangles)
