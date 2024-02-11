import warp as wp

class mat99d(wp.types.matrix(shape=(9, 9), dtype=wp.float64)):
    pass

@wp.func
def cofactor(F: wp.mat33d) -> wp.mat33d:
    JFInvT = wp.mat33d()
    JFInvT[0, 0] = F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]
    JFInvT[0, 1] = F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2]
    JFInvT[0, 2] = F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]
    JFInvT[1, 0] = F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2]
    JFInvT[1, 1] = F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0]
    JFInvT[1, 2] = F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1]
    JFInvT[2, 0] = F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]
    JFInvT[2, 1] = F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]
    JFInvT[2, 2] = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    return JFInvT

@wp.func
def make_pd(A: wp.mat33d) -> wp.mat33d:
    Q = wp.mat33d()
    d = wp.vec3d()
    wp.eig3(A, Q, d)
    for i in range(3):
        d[i] = wp.max(d[i], wp.float64(0))
    return Q @ wp.diag(d) @ wp.transpose(Q)

@wp.func
def make_pd(A: wp.mat22d) -> wp.mat22d:
    a = A[0, 0]
    b = (A[0, 1] + A[1, 0]) / wp.float64(2.0)
    d = A[1, 1]
    b2 = b * b
    D = a * d - b2
    T_div_2 = (a + d) / wp.float64(2.0)
    sqrtTT4D = wp.sqrt(T_div_2 * T_div_2 - D)
    L2 = T_div_2 - sqrtTT4D
    zero = wp.float64(0.0)
    symMtr = wp.mat22d(zero, zero, zero, zero)
    if L2 < zero:
        L1 = T_div_2 + sqrtTT4D
        if L1 <= zero:
            symMtr = wp.mat22d(zero, zero, zero, zero)
        else:
            if b2 == zero:
                symMtr = wp.mat22d(zero, zero, zero, zero)
                symMtr[0, 0] = L1
            else:
                L1md = L1 - d
                L1md_div_L1 = L1md / L1
                symMtr[0, 0] = L1md_div_L1 * L1md
                symMtr[0, 1] = b * L1md_div_L1
                symMtr[1, 0] = b * L1md_div_L1
                symMtr[1, 1] = b2 / L1
    return symMtr

