import numpy as _N
cimport numpy as _N
cimport cython
import time as _tm

"""
c functions
"""
#cdef extern from "stdio.h":
#    double sqrt(double)


def show(double[:, :, ::1] A, int D1, int D2, int D3):
    cdef int i, j, k, iD2D3, jD3
    cdef double *p_A   = &A[0, 0, 0]

    Mat = _N.empty((D1, D2, D3))
    cdef double[:, :, ::1] Matmv = Mat   #  memory view
    cdef double *pMat    = &Matmv[0, 0, 0]

    for 0 <= i < D1:
        iD2D3 = i*D2*D3
        for 0 <= j < D2:
            jD3 = j*D3
            for 0 <= k < D3:
                print "%.3f" % p_A[iD2D3 + jD3 + k]
                pMat[iD2D3 + jD3 + k] = p_A[iD2D3 + jD3 + k]

    return Mat
