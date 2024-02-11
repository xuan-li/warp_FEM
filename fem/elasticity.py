from .neohookean import neohookean_energy, neohookean_grad, neohookean_hessian
from .utils import mat99d
import warp as wp

class mat12_9d(wp.types.matrix(shape=(12, 9), dtype=wp.float64)):
    pass

class mat12_12d(wp.types.matrix(shape=(12, 12), dtype=wp.float64)):
    pass

class vec12d(wp.types.vector(length=12, dtype=wp.float64)):
    pass

@wp.func
def get_deformation_gradient(v0: wp.vec3d, v1:wp.vec3d, v2: wp.vec3d, v3: wp.vec3d, IB: wp.mat33d) -> wp.mat33d:
    T = wp.mat33d()
    for d in range(3):
        T[d, 0] = v1[d] - v0[d]
        T[d, 1] = v2[d] - v0[d]
        T[d, 2] = v3[d] - v0[d]
    return T @ IB

@wp.func
def backpropagate_element_gradient(IB:wp.mat33d, de_dF: wp.mat33d) -> vec12d:
    de_dX = vec12d(wp.float64(0.))
    R10 = IB[0, 0] * de_dF[0, 0] + IB[0, 1] * de_dF[0, 1] + IB[0, 2] * de_dF[0, 2]
    R11 = IB[0, 0] * de_dF[1, 0] + IB[0, 1] * de_dF[1, 1] + IB[0, 2] * de_dF[1, 2]
    R12 = IB[0, 0] * de_dF[2, 0] + IB[0, 1] * de_dF[2, 1] + IB[0, 2] * de_dF[2, 2]
    R20 = IB[1, 0] * de_dF[0, 0] + IB[1, 1] * de_dF[0, 1] + IB[1, 2] * de_dF[0, 2]
    R21 = IB[1, 0] * de_dF[1, 0] + IB[1, 1] * de_dF[1, 1] + IB[1, 2] * de_dF[1, 2]
    R22 = IB[1, 0] * de_dF[2, 0] + IB[1, 1] * de_dF[2, 1] + IB[1, 2] * de_dF[2, 2]
    R30 = IB[2, 0] * de_dF[0, 0] + IB[2, 1] * de_dF[0, 1] + IB[2, 2] * de_dF[0, 2]
    R31 = IB[2, 0] * de_dF[1, 0] + IB[2, 1] * de_dF[1, 1] + IB[2, 2] * de_dF[1, 2]
    R32 = IB[2, 0] * de_dF[2, 0] + IB[2, 1] * de_dF[2, 1] + IB[2, 2] * de_dF[2, 2]
    de_dX[1 * 3 + 0] = R10
    de_dX[1 * 3 + 1] = R11
    de_dX[1 * 3 + 2] = R12
    de_dX[2 * 3 + 0] = R20
    de_dX[2 * 3 + 1] = R21
    de_dX[2 * 3 + 2] = R22
    de_dX[3 * 3 + 0] = R30
    de_dX[3 * 3 + 1] = R31
    de_dX[3 * 3 + 2] = R32
    de_dX[0 * 3 + 0] = -R10 - R20 - R30
    de_dX[0 * 3 + 1] = -R11 - R21 - R31
    de_dX[0 * 3 + 2] = -R12 - R22 - R32
    return de_dX

@wp.func
def backpropagate_element_hessian(IB: wp.mat33d, d2e_dF2: mat99d) -> mat12_12d:
    intermediate = mat12_9d(wp.float64(0.))
    for colI in range(9):
        intermediate[3, colI] = IB[0, 0] * d2e_dF2[0, colI] + IB[0, 1] * d2e_dF2[3, colI] + IB[0, 2] * d2e_dF2[6, colI]
        intermediate[4, colI] = IB[0, 0] * d2e_dF2[1, colI] + IB[0, 1] * d2e_dF2[4, colI] + IB[0, 2] * d2e_dF2[7, colI]
        intermediate[5, colI] = IB[0, 0] * d2e_dF2[2, colI] + IB[0, 1] * d2e_dF2[5, colI] + IB[0, 2] * d2e_dF2[8, colI]
        intermediate[6, colI] = IB[1, 0] * d2e_dF2[0, colI] + IB[1, 1] * d2e_dF2[3, colI] + IB[1, 2] * d2e_dF2[6, colI]
        intermediate[7, colI] = IB[1, 0] * d2e_dF2[1, colI] + IB[1, 1] * d2e_dF2[4, colI] + IB[1, 2] * d2e_dF2[7, colI]
        intermediate[8, colI] = IB[1, 0] * d2e_dF2[2, colI] + IB[1, 1] * d2e_dF2[5, colI] + IB[1, 2] * d2e_dF2[8, colI]
        intermediate[9, colI] = IB[2, 0] * d2e_dF2[0, colI] + IB[2, 1] * d2e_dF2[3, colI] + IB[2, 2] * d2e_dF2[6, colI]
        intermediate[10, colI] = IB[2, 0] * d2e_dF2[1, colI] + IB[2, 1] * d2e_dF2[4, colI] + IB[2, 2] * d2e_dF2[7, colI]
        intermediate[11, colI] = IB[2, 0] * d2e_dF2[2, colI] + IB[2, 1] * d2e_dF2[5, colI] + IB[2, 2] * d2e_dF2[8, colI]
        intermediate[0, colI] = -intermediate[3, colI] - intermediate[6, colI] - intermediate[9, colI]
        intermediate[1, colI] = -intermediate[4, colI] - intermediate[7, colI] - intermediate[10, colI]
        intermediate[2, colI] = -intermediate[5, colI] - intermediate[8, colI] - intermediate[11, colI]

    d2e_dX2 = mat12_12d(wp.float64(0.0))
    for rowI in range(12):
        _000 = IB[0, 0] * intermediate[rowI, 0]
        _013 = IB[0, 1] * intermediate[rowI, 3]
        _026 = IB[0, 2] * intermediate[rowI, 6]
        _001 = IB[0, 0] * intermediate[rowI, 1]
        _014 = IB[0, 1] * intermediate[rowI, 4]
        _027 = IB[0, 2] * intermediate[rowI, 7]
        _002 = IB[0, 0] * intermediate[rowI, 2]
        _015 = IB[0, 1] * intermediate[rowI, 5]
        _028 = IB[0, 2] * intermediate[rowI, 8]
        _100 = IB[1, 0] * intermediate[rowI, 0]
        _113 = IB[1, 1] * intermediate[rowI, 3]
        _126 = IB[1, 2] * intermediate[rowI, 6]
        _101 = IB[1, 0] * intermediate[rowI, 1]
        _114 = IB[1, 1] * intermediate[rowI, 4]
        _127 = IB[1, 2] * intermediate[rowI, 7]
        _102 = IB[1, 0] * intermediate[rowI, 2]
        _115 = IB[1, 1] * intermediate[rowI, 5]
        _128 = IB[1, 2] * intermediate[rowI, 8]
        _200 = IB[2, 0] * intermediate[rowI, 0]
        _213 = IB[2, 1] * intermediate[rowI, 3]
        _226 = IB[2, 2] * intermediate[rowI, 6]
        _201 = IB[2, 0] * intermediate[rowI, 1]
        _214 = IB[2, 1] * intermediate[rowI, 4]
        _227 = IB[2, 2] * intermediate[rowI, 7]
        _202 = IB[2, 0] * intermediate[rowI, 2]
        _215 = IB[2, 1] * intermediate[rowI, 5]
        _228 = IB[2, 2] * intermediate[rowI, 8]
        d2e_dX2[rowI, 3] = _000 + _013 + _026
        d2e_dX2[rowI, 4] = _001 + _014 + _027
        d2e_dX2[rowI, 5] = _002 + _015 + _028
        d2e_dX2[rowI, 6] = _100 + _113 + _126
        d2e_dX2[rowI, 7] = _101 + _114 + _127
        d2e_dX2[rowI, 8] = _102 + _115 + _128
        d2e_dX2[rowI, 9] = _200 + _213 + _226
        d2e_dX2[rowI, 10] = _201 + _214 + _227
        d2e_dX2[rowI, 11] = _202 + _215 + _228
        d2e_dX2[rowI, 0] = -_200 - _213 - _226 - _100 - _113 - _126 - _000 - _013 - _026
        d2e_dX2[rowI, 1] = -_001 - _014 - _027 - _101 - _114 - _127 - _201 - _214 - _227
        d2e_dX2[rowI, 2] = -_002 - _015 - _028 - _102 - _115 - _128 - _202 - _215 - _228
    
    return d2e_dX2

@wp.kernel
def compute_elasticity_energy(x: wp.array(dtype=wp.vec3d), tet: wp.array(dtype=wp.vec4i), IB: wp.array(dtype=wp.mat33d), vol: wp.array(dtype=wp.float64), mu: wp.array(dtype=wp.float64), lam: wp.array(dtype=wp.float64), scale: wp.float64, energy: wp.array(dtype=wp.float64)):
    tid = wp.tid()
    index = tet[tid]
    v0 = x[index[0]]
    v1 = x[index[1]]
    v2 = x[index[2]]
    v3 = x[index[3]]
    F = get_deformation_gradient(v0, v1, v2, v3, IB[tid])
    energy[tid] = energy[tid] + scale * vol[tid] * neohookean_energy(F, mu[tid], lam[tid])

@wp.kernel
def compute_elasticity_grad(x: wp.array(dtype=wp.vec3d), tet: wp.array(dtype=wp.vec4i), IB: wp.array(dtype=wp.mat33d), vol: wp.array(dtype=wp.float64), mu: wp.array(dtype=wp.float64), lam: wp.array(dtype=wp.float64), scale: wp.float64, grad: wp.array(dtype=wp.vec3d)):
    tid = wp.tid()
    index = tet[tid]
    v0 = x[index[0]]
    v1 = x[index[1]]
    v2 = x[index[2]]
    v3 = x[index[3]]
    F = get_deformation_gradient(v0, v1, v2, v3, IB[tid])
    de_dF = neohookean_grad(F, mu[tid], lam[tid])
    de_dX = backpropagate_element_gradient(IB[tid], de_dF)
    indices = tet[tid]
    for i in range(4):
        idx = indices[i]
        wp.atomic_add(grad, idx, scale * vol[tid] * wp.vec3d(de_dX[i * 3 + 0], de_dX[i * 3 + 1], de_dX[i * 3 + 2]))

@wp.kernel
def compute_elasticity_hessian(x: wp.array(dtype=wp.vec3d), 
                               tet: wp.array(dtype=wp.vec4i), 
                               IB: wp.array(dtype=wp.mat33d), 
                               vol: wp.array(dtype=wp.float64), 
                               mu: wp.array(dtype=wp.float64), 
                               lam: wp.array(dtype=wp.float64), 
                               offset: wp.int32, 
                               rows: wp.array(dtype=wp.int32), 
                               cols:wp.array(dtype=wp.int32), 
                               values: wp.array(dtype=wp.mat33d), 
                               scale: wp.float64, 
                               project_pd: wp.int32):
    tid = wp.tid()
    index = tet[tid]
    v0 = x[index[0]]
    v1 = x[index[1]]
    v2 = x[index[2]]
    v3 = x[index[3]]
    F = get_deformation_gradient(v0, v1, v2, v3, IB[tid])
    dPdF = neohookean_hessian(F, mu[tid], lam[tid], project_pd)
    d2e_dX2 = backpropagate_element_hessian(IB[tid], dPdF)
    indices = tet[tid]
    for i in range(4):
        for j in range(4):
            row = indices[i]
            col = indices[j]
            rows[offset + tid * 16 + i * 4 + j] = row
            cols[offset + tid * 16 + i * 4 + j] = col
            v = wp.mat33d(wp.float64(0))
            for k in range(3):
                for l in range(3):
                    v[k, l] = scale * vol[tid] * d2e_dX2[i * 3 + k, j * 3 + l]
            values[offset + tid * 16 + i * 4 + j] = v

    