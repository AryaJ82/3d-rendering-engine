import numpy
from math import cos, sin, pi

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# def function
mod = SourceModule("""
__device__ void rotate_vector(float* v, float x, float y, float z, float* rot)
{
    // Get rid of v as a pointer. Just write to *dest directly
    v[0] = x;
    v[1] = y;
    v[2] = z;
    y = v[1] * rot[0] + v[2] * rot[1];
    z = v[2] * rot[0] - v[1] * rot[1];

    v[0] = x;
    v[1] = y;
    v[2] = z;
    x = v[0] * rot[2] - v[2] * rot[3];
    z = v[0] * rot[3] + v[2] * rot[2];

    v[0] = x;
    v[1] = y;
    v[2] = z;
    x = v[0] * rot[4] + v[1] * rot[5];
    y = v[1] * rot[4] - v[0] * rot[5];

    v[0] = x;
    v[1] = y;
    v[2] = z;
}

__device__ void matrix_mult(float* v, float x, float y, float z, float* matrix)
{
    // Get rid of v as a pointer. Just write to *dest directly
    // change matrix to be in normal form (transverse of what it is now)
    v[0] = x * matrix[0] + y * matrix[1] + z * matrix[2] + matrix[3];
    v[1] = x * matrix[4] + y * matrix[5] + z * matrix[6] + matrix[7];
    v[2] = x * matrix[8] + y * matrix[9] + z * matrix[10] + matrix[11];

}

__device__ void norm_mmult(float* v, float x, float y, float z, float* matrix)
{
    // Get rid of v as a pointer. Just write to *dest directly
    // change matrix to be in normal form (transverse of what it is now)
    v[0] = x * matrix[0] + y * matrix[1] + z * matrix[2];
    v[1] = x * matrix[4] + y * matrix[5] + z * matrix[6];
    v[2] = x * matrix[8] + y * matrix[9] + z * matrix[10];

}

__device__ void project(float* v, float* mesh_ro,  float* proj_mat)
{
    float z = v[2] + mesh_ro[2];
    matrix_mult(v, v[0] + mesh_ro[0], v[1] + mesh_ro[1], v[2] + mesh_ro[2], proj_mat);
    

    if (z != 0) {
        v[0] /= z;
        v[0] += 1;
        v[0] /= 2;
        
        v[1] /= z;
        v[1] += 1;
        v[1] /= 2;
    }
    v[0] *= 800;
    v[1] *= 800;
}

__global__ void raster(float* dest, float* triangles, float* mesh_ro, float* rot, float* view_mat, float* proj_mat)
{
    int idx = threadIdx.x * 12 + threadIdx.y * 3;

    float v[3];
    // Rotate vectors
    rotate_vector(v, triangles[idx], triangles[idx+1], triangles[idx+2], rot);


    // View transform vector and project vectors
    if (idx % 4 == 0) // This vector is a normal. We must use a diff operation
    {
        norm_mmult(v, v[0], v[1], v[2], view_mat);
        // normal should not be projected or have its origin moved

    }
    else {
        //matrix_mult(v, triangles[idx], triangles[idx + 1], triangles[idx + 2], view_mat);
        matrix_mult(v, v[0], v[1], v[2], view_mat);
        project(v, mesh_ro, proj_mat);

    }

    dest[idx] = v[0];
    dest[idx + 1] = v[1];
    dest[idx + 2] = v[2];
}
  """)

raster = mod.get_function("raster")
def CUDA_raster(ta, mesh_ro, rot, view_mat, proj_mat):
    """
    Returns triange_array as an projected array of triangles ready to be drawn
    to the screen.

    :param triangle_array:
    :param mesh_ro:
    :param rot:
    :param view_mat:
    :param proj_mat:
    :return: List[List[float]]
    """
    # TODO: below can be modified if i fully transition to numpy as my datatype
    ta_rastered = numpy.empty_like(ta)
    raster(
        cuda.Out(ta_rastered),
        cuda.In(ta),
        cuda.In(numpy.array(mesh_ro, dtype=numpy.float32)),
        cuda.In(numpy.array(rot, dtype=numpy.float32)),
        cuda.In(numpy.array(view_mat, dtype=numpy.float32)),
        cuda.In(numpy.array(proj_mat, dtype=numpy.float32)),
        block=(len(ta), 3+1, 1))
    return ta_rastered
