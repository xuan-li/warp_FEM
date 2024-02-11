import warp as wp
from .utils import make_pd, cofactor, mat99d

@wp.func
def neohookean_energy(F: wp.mat33d, mu: wp.float64, lam: wp.float64):
    I1 = wp.trace(wp.transpose(F) @ F)
    J = wp.determinant(F)
    if J < 0:
        return wp.float64(1e10)
    else:
        return mu / wp.float64(2.) * (I1 - wp.float64(3.)) - mu * wp.log(J) + lam / wp.float64(2.) * wp.log(J) ** wp.float64(2.)

@wp.func
def neohookean_grad(F: wp.mat33d, mu: wp.float64, lam: wp.float64) -> wp.mat33d:
    J = wp.determinant(F)
    FInvT = cofactor(F) / J
    return mu * (F - FInvT) + lam * wp.log(J) * FInvT

@wp.func
def singular_grad(sigma: wp.vec3d, mu: wp.float64, lam: wp.float64) -> wp.vec3d:
    sigmaProd = sigma[0] * sigma[1] * sigma[2]
    log_sigmaProd = wp.log(sigmaProd)
    one = wp.float64(1.)
    inv0 = one / sigma[0]
    inv1 = one / sigma[1]
    inv2 = one / sigma[2]
    return wp.vec3d(mu * (sigma[0] - inv0) + lam * inv0 * log_sigmaProd,
                    mu * (sigma[1] - inv1) + lam * inv1 * log_sigmaProd,
                    mu * (sigma[2] - inv2) + lam * inv2 * log_sigmaProd)

@wp.func
def singular_hessian(sigma: wp.vec3d, mu: wp.float64, lam: wp.float64) -> wp.mat33d:
    sigmaProd = sigma[0] * sigma[1] * sigma[2]
    log_sigmaProd = wp.log(sigmaProd)
    one = wp.float64(1.)
    inv2_0 = one / sigma[0] / sigma[0]
    inv2_1 = one / sigma[1] / sigma[1]
    inv2_2 = one / sigma[2] / sigma[2]
    H00 = mu * (one + inv2_0) - lam * inv2_0 * (log_sigmaProd - one)
    H11 = mu * (one + inv2_1) - lam * inv2_1 * (log_sigmaProd - one)
    H22 = mu * (one + inv2_2) - lam * inv2_2 * (log_sigmaProd - one)
    H01 = lam / sigma[0] / sigma[1]
    H12 = lam / sigma[1] / sigma[2]
    H02 = lam / sigma[0] / sigma[2]
    return wp.mat33d(H00, H01, H02, H01, H11, H12, H02, H12, H22)


@wp.func
def neohookean_hessian(F: wp.mat33d, mu: wp.float64, lam:wp.float64, project_pd:wp.int32) -> mat99d:
    # Caution!!! F is flattened in column-major manner. which is not aligned with torch/warp reprentation
    U = wp.mat33d()
    V = wp.mat33d()
    sigma = wp.vec3d()
    wp.svd3(F, U, sigma, V)
    sigmaProd = sigma[0] * sigma[1] * sigma[2]
    dE_div_dsigma = singular_grad(sigma, mu, lam)
    d2E_div_dsigma2 = singular_hessian(sigma, mu, lam)

    middle = mu - lam * wp.log(sigmaProd)
    leftCoef = (mu + middle / sigma[0] / sigma[1]) / wp.float64(2.)
    rightCoef = (dE_div_dsigma[0] + dE_div_dsigma[1]) / (sigma[0] + sigma[1]) / wp.float64(2.)
    B0 = wp.mat22d(leftCoef + rightCoef, leftCoef - rightCoef, leftCoef - rightCoef, leftCoef + rightCoef)

    leftCoef = (mu + middle / sigma[1] / sigma[2]) / wp.float64(2.)
    rightCoef = (dE_div_dsigma[1] + dE_div_dsigma[2]) / (sigma[1] + sigma[2]) / wp.float64(2.)
    B1 = wp.mat22d(leftCoef + rightCoef, leftCoef - rightCoef, leftCoef - rightCoef, leftCoef + rightCoef)

    leftCoef = (mu + middle / sigma[2] / sigma[0]) / wp.float64(2.)
    rightCoef = (dE_div_dsigma[2] + dE_div_dsigma[0]) / (sigma[2] + sigma[0]) / wp.float64(2.)
    B2 = wp.mat22d(leftCoef + rightCoef, leftCoef - rightCoef, leftCoef - rightCoef, leftCoef + rightCoef)

    if project_pd:
        d2E_div_dsigma2 = make_pd(d2E_div_dsigma2)
        B0 = make_pd(B0)
        B1 = make_pd(B1)
        B2 = make_pd(B2)

    M = mat99d(wp.float64(0.))
    dPdF = mat99d(wp.float64(0.))
    M[0, 0] = d2E_div_dsigma2[0, 0]
    M[0, 4] = d2E_div_dsigma2[0, 1]
    M[0, 8] = d2E_div_dsigma2[0, 2]
    M[4, 0] = d2E_div_dsigma2[1, 0]
    M[4, 4] = d2E_div_dsigma2[1, 1]
    M[4, 8] = d2E_div_dsigma2[1, 2]
    M[8, 0] = d2E_div_dsigma2[2, 0]
    M[8, 4] = d2E_div_dsigma2[2, 1]
    M[8, 8] = d2E_div_dsigma2[2, 2]
    M[1, 1] = B0[0, 0]
    M[1, 3] = B0[0, 1]
    M[3, 1] = B0[1, 0]
    M[3, 3] = B0[1, 1]
    M[5, 5] = B1[0, 0]
    M[5, 7] = B1[0, 1]
    M[7, 5] = B1[1, 0]
    M[7, 7] = B1[1, 1]
    M[2, 2] = B2[1, 1]
    M[2, 6] = B2[1, 0]
    M[6, 2] = B2[0, 1]
    M[6, 6] = B2[0, 0]
    for j in range(3):
        for i in range(3):
            for s in range(3):
                for r in range(3):
                    ij = j * 3 + i
                    rs = s * 3 + r
                    dPdF[ij, rs] =    M[0, 0] * U[i, 0] * V[j, 0] * U[r, 0] * V[s, 0] \
                                    + M[0, 4] * U[i, 0] * V[j, 0] * U[r, 1] * V[s, 1] \
                                    + M[0, 8] * U[i, 0] * V[j, 0] * U[r, 2] * V[s, 2] \
                                    + M[4, 0] * U[i, 1] * V[j, 1] * U[r, 0] * V[s, 0] \
                                    + M[4, 4] * U[i, 1] * V[j, 1] * U[r, 1] * V[s, 1] \
                                    + M[4, 8] * U[i, 1] * V[j, 1] * U[r, 2] * V[s, 2] \
                                    + M[8, 0] * U[i, 2] * V[j, 2] * U[r, 0] * V[s, 0] \
                                    + M[8, 4] * U[i, 2] * V[j, 2] * U[r, 1] * V[s, 1] \
                                    + M[8, 8] * U[i, 2] * V[j, 2] * U[r, 2] * V[s, 2] \
                                    + M[1, 1] * U[i, 0] * V[j, 1] * U[r, 0] * V[s, 1] \
                                    + M[1, 3] * U[i, 0] * V[j, 1] * U[r, 1] * V[s, 0] \
                                    + M[3, 1] * U[i, 1] * V[j, 0] * U[r, 0] * V[s, 1] \
                                    + M[3, 3] * U[i, 1] * V[j, 0] * U[r, 1] * V[s, 0] \
                                    + M[5, 5] * U[i, 1] * V[j, 2] * U[r, 1] * V[s, 2] \
                                    + M[5, 7] * U[i, 1] * V[j, 2] * U[r, 2] * V[s, 1] \
                                    + M[7, 5] * U[i, 2] * V[j, 1] * U[r, 1] * V[s, 2] \
                                    + M[7, 7] * U[i, 2] * V[j, 1] * U[r, 2] * V[s, 1] \
                                    + M[2, 2] * U[i, 0] * V[j, 2] * U[r, 0] * V[s, 2] \
                                    + M[2, 6] * U[i, 0] * V[j, 2] * U[r, 2] * V[s, 0] \
                                    + M[6, 2] * U[i, 2] * V[j, 0] * U[r, 0] * V[s, 2] \
                                    + M[6, 6] * U[i, 2] * V[j, 0] * U[r, 2] * V[s, 0]
    return dPdF


@wp.kernel
def test_full(F: wp.array(dtype=wp.mat33d), energy: wp.array(dtype=wp.float64), grad: wp.array(dtype=wp.mat33d), hess: wp.array(dtype=mat99d)):
    tid = wp.tid()
    one = wp.float64(1.)
    zero = wp.float64(0.)
    energy[tid] = neohookean_energy(F[tid], one, one)
    grad[tid] = neohookean_grad(F[tid], one, one)
    hess[tid] = neohookean_hessian(F[tid], one, one, 0)


@wp.kernel
def test_singular(sigma: wp.array(dtype=wp.vec3d), grad: wp.array(dtype=wp.vec3d), hess: wp.array(dtype=wp.mat33d)):
    tid = wp.tid()
    one = wp.float64(1.)
    grad[tid] = singular_grad(sigma[tid], one, one)
    hess[tid] = singular_hessian(sigma[tid], one, one)


if __name__ == "__main__":
    wp.init()
    import torch
    torch.set_default_dtype(torch.float64)

    # test sigma derivatives
    sigma = torch.asarray([[0.9, 0.8, 0.7]], dtype=torch.float64).cuda()
    eps = 1e-5
    delta = torch.rand_like(sigma)
    delta /= torch.linalg.norm(delta.reshape(sigma.shape[0], -1), axis=-1)[:, None]
    sigma1 = sigma + delta * eps
    sigma2 = sigma - delta * eps
    sigma = torch.cat([sigma1, sigma2], axis=0).contiguous()
    grad = torch.zeros_like(sigma)
    hess = torch.zeros([sigma.shape[0], 3, 3]).cuda()

    wp.launch(kernel=test_singular, dim=2, 
              inputs=[wp.from_torch(sigma, dtype=wp.vec3d),
                      wp.from_torch(grad, dtype=wp.vec3d), 
                      wp.from_torch(hess, dtype=wp.mat33d)], 
              device="cuda")
    
    sigma = sigma.reshape((sigma.shape[0], -1))
    delta = delta.flatten()
    hess_eps = torch.linalg.norm((grad[0] - grad[1]) / eps - ((hess[0] + hess[1]) @ delta[:, None]).flatten(), axis=-1).item()
    print(hess_eps)

    # test full derivatives
    F = torch.asarray([[[0.3, 0.1, 0], [0, 1.1, 0], [0.0, 0, 0.5]]], dtype=torch.float64).cuda()
    eps = 1e-5
    delta = torch.rand_like(F)
    delta /= torch.linalg.norm(delta.reshape(F.shape[0], -1), axis=-1)[:, None, None]
    F1 = F + delta * eps
    F2 = F - delta * eps
    F = torch.cat([F1, F2], axis=0).contiguous()
    energy = torch.zeros([F.shape[0], 1]).cuda()
    grad = torch.zeros_like(F)
    hess = torch.zeros([F.shape[0], 9, 9]).cuda()

    wp.launch(kernel=test_full, dim=2, 
              inputs=[wp.from_torch(F, dtype=wp.mat33d),
                      wp.from_torch(energy.flatten(), dtype=wp.float64),
                      wp.from_torch(grad, dtype=wp.mat33d), 
                      wp.from_torch(hess, dtype=mat99d)], 
              device="cuda")
    
    F = F.transpose(1, 2).reshape((F.shape[0], -1))
    grad = grad.transpose(1, 2).reshape((F.shape[0], -1))
    delta = delta.transpose(1, 2).flatten()
    grad_eps = abs(((energy[0] - energy[1]) / eps - torch.sum(delta * (grad[0] + grad[1])))).item()
    hess_eps = torch.linalg.norm((grad[0] - grad[1]) / eps - ((hess[0] + hess[1]) @ delta[:, None]).flatten(), axis=-1).item()
    print(grad_eps, hess_eps)