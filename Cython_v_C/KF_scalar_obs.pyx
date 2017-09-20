#  k-dimensional state space
#  N steps  length of observation
#  1-step prediction px  N x k
#  filter            fx  N x k
#  H                 linear comb of state space to arrive at (noiseless) obsevation

N  = 1000
k  = 9

px   = _N.random.randn(N, k)
fx   = _N.empty((N, k))
H    = _N.zeros(k)
H[0] = 1
K    = _N.empty

#  The mean in case of scalar observation y,
#  is xf[n] = xf[n-1] + K[n](y[n] - Hx


