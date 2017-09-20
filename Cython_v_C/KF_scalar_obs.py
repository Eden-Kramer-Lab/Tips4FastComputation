import time as _tm
import cython_v as c_v

#  k-dimensional state space
#  N steps  length of observation
#  1-step prediction px  N x k
#  filter            fx  N x k
#  H                 linear comb of state space to arrive at (noiseless) obsevation

N  = 1000
k  = 9

px   = _N.random.randn(N, k, 1)
fx   = _N.empty((N, k, 1))
H    = _N.zeros((1, k))
K    = _N.random.randn(N, k, 1)
y    = _N.random.randn(N)
KyHpx = _N.empty((k, 1))

fx_cp= _N.empty((N, k, 1))   #  fx and fx_cp should be same answer


#################  Fast implementation.  Cython
t1   = _tm.time()
c_v.FFmean(k, N, K, y, px, fx_cp)
t2   = _tm.time()

print (t2-t1)

#################  Slow implementation.  Numpy
t3   = _tm.time()
for n in xrange(N):
    _N.multiply(K[n], y[n] - px[n, 0, 0], out=KyHpx)
    _N.add(px[n], KyHpx, out=fx[n])
t4   = _tm.time()

print (t4-t3)



print (t4-t3)/(t2-t1)

