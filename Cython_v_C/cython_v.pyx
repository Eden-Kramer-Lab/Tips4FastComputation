cimport cython
import numpy as _N
cimport numpy as _N

@cython.boundscheck(False)
@cython.wraparound(False)
def FF_mean(long k, long N, double[:, :, ::1] K_mv, double[::1] y_mv, double[:, :, ::1] px_mv, double[:, :, ::1] fx_mv):
    #  Forward filter mean
    cdef long n, i, j, nk

    cdef double *p_fx = &fx_mv[0, 0, 0]
    cdef double *p_px = &px_mv[0, 0, 0]
    cdef double *p_y  = &y_mv[0]
    cdef double *p_K  = &K_mv[0, 0, 0]

    KyHpx = _N.empty((k, 1))
    cdef double[:, ::1] KyHpx_mv = KyHpx
    cdef double *p_KyHpx = &KyHpx_mv[0, 0]

    with nogil:
        for 0 <= n < N:
            nk = n*k

            for 0 <= j < k:
                p_KyHpx[j] = p_K[nk+j] * (p_y[n] - p_px[nk])

            for 0 <= j < k:
                p_fx[nk+j] = p_px[nk+j] + p_KyHpx[j]
