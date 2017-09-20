#exf("../pyscripts/kflib.py")
import time as _tm

"""
We want to draw N k-dim multivariate random numbers.
All N of them have different means and covariances.  

Numpy provides a good implementation for case where we want to draw N realizations of a k-dim multivariate normal, but doesn't provide an implementation of drawing N times, 1 realization each of N multivariate normals with unique means and covariances.

In this case, Cholesky decomposition of N covariance matrices is the fastest way to go.
"""
def covMat(dim, cc=1):
    cm   = _N.empty((dim, dim))
    stds = _N.abs(_N.random.randn(dim))
    pij  = _N.random.rand(dim, dim) * cc   #  use upper half only

    _N.fill_diagonal(cm, _N.abs(1 + 0.3*_N.random.randn(dim)))
    for i in xrange(dim):
        for j in xrange(i+1, dim):
            cm[i, j] = _N.sqrt(cm[i, i]*cm[j,j])*cc*(_N.random.rand()-0.5)*2
            cm[j, i] = cm[i, j]
    return cm

def numpyMVN0(us, Sgs, N, k):
    #  sample N w/ same cov

    t1     = _tm.time()
    rn0    = _N.empty((N, k))
    rn0 = us + _N.random.multivariate_normal(_N.zeros(k), Sg[n], size=N)
    t2     = _tm.time()
    return rn0, (t2-t1)

def numpyMVN(us, Sgs, N, k):
    t1     = _tm.time()
    rn1    = _N.empty((N, k))
    for n in xrange(N):
        rn1[n] = us[n] + _N.random.multivariate_normal(_N.zeros(k), Sg[n], size=1)
    t2     = _tm.time()
    return rn1, (t2-t1)

def mvnSVDa(us, Sgs, N, k):
    t1     = _tm.time()
    Ik     = _N.identity(k)
    rn2    = _N.random.randn(N, k)
    #rn2    = _N.random.multivariate_normal(_N.zeros(k), Ik, size=N)
    S,V,D  = _N.linalg.svd(Sgs)
    Vs     = _N.sqrt(V)

    rn2a   = _N.empty((N, k))
    for n in xrange(N):
        rn2a[n] = _N.dot(S[n], Vs[n, :]*rn2[n, :])
    t2     = _tm.time()
    return rn2a, (t2-t1)

def mvnSVDb(us, Sgs, N, k):
    t1     = _tm.time()
    Ik     = _N.identity(k)
    rn2    = _N.random.randn(N, k)
    #rn2    = _N.random.multivariate_normal(_N.zeros(k), Ik, size=N)
    S,V,D  = _N.linalg.svd(Sgs)
    Vs     = _N.sqrt(V)
    VsRn2b = Vs*rn2
    rn2b   = us + _N.einsum("njk,nk->nj", S, VsRn2b)
    t2     = _tm.time()
    return rn2b, (t2-t1)

def mvnChol_a(us, Sgs, N, k):
    t1     = _tm.time()
    Ik     = _N.identity(k)
    rn3    = _N.random.multivariate_normal(_N.zeros(k), Ik, size=N)
    C      = _N.linalg.cholesky(Sgs)
    rn3a   = us + _N.einsum("njk,nk->nj", C, rn3)
    t2     = _tm.time()
    return rn3a, (t2-t1)

def mvnChol_b(us, Sgs, N, k):
    t1     = _tm.time()
    C      = _N.linalg.cholesky(Sgs)
    rn3    = _N.random.randn(N, k)
    rn3b   = us + _N.einsum("njk,nk->nj", C, rn3)
    t2     = _tm.time()
    return rn3b, (t2-t1)

    

N      = 1000          #  number of multivar norms to draw
k      = 9
us     = _N.zeros((N, k))
Sg     = _N.empty((N, k, k))
Sg0    = covMat(k, cc=0.2)

for n in xrange(N):
    Sg[n]    = Sg0

testIDs = _N.arange(6)
_N.random.shuffle(testIDs)

for tid in testIDs:
    #############################   Numpy mvn
    if tid == 0:
        rn0, dt0     = numpyMVN0(us, Sg, N, k)
    if tid == 1:
        rn1, dt1     = numpyMVN(us, Sg, N, k)
    #############################   SVD
    elif tid == 2:
        rn2a, dt2a   = mvnSVDa(us, Sg, N, k)
    elif tid == 3:
        rn2b, dt2b   = mvnSVDb(us, Sg, N, k)
    #############################   Cholesky
    elif tid == 4:
        rn3a, dt3a   = mvnChol_a(us, Sg, N, k)
    elif tid == 5:
        rn3b, dt3b   = mvnChol_b(us, Sg, N, k)
#############################

print "Naive0  %.3e" % dt0
print "Naive  %.3e" % dt1
print "SVD1   %.3e" % dt2a
print "SVD2   %.3e" % dt2b
print "CHO1   %.3e" % dt3a
print "CHO2   %.3e" % dt3b

"""
S0, V0, D0 = _N.linalg.svd(Sg[0])
C0       = _N.linalg.cholesky(Sg[0])
print _N.dot(S0, _N.dot(_N.diag(V0), D0))
print _N.dot(C0, C0.T)

_N.dot(S0, _N.sqrt(V0)*rn2[0])
"""
_N.cov(rn3a, rowvar=0)
