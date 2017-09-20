import cy_ver as _cyv

D1 = 3
D2 = 2
D3 = 4
A = _N.random.randn(D1, D2, D3)

for i in xrange(D1):
    for j in xrange(D2):
        for k in xrange(D3):
            print "%.3f" % A[i, j, k]

print "----------------"
pM = _cyv.show(A, D1, D2, D3)
