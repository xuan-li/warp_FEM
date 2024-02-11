import warp as wp
import warp.sparse as wps
from warp.utils import array_sum, array_inner

@wp.kernel
def array_inv(x: wp.array(dtype=wp.mat33d)):
    tid = wp.tid()
    x[tid] = wp.inverse(x[tid])

@wp.kernel
def array_matmul(A: wp.array(dtype=wp.mat33d), y: wp.array(dtype=wp.vec3d), z: wp.array(dtype=wp.vec3d)):
    tid = wp.tid()
    z[tid] = A[tid] @ y[tid]

# @wp.kernel
# def array_dot(x: wp.array(dtype=wp.vec3d), y : wp.array(dtype=wp.vec3d), z: wp.array(dtype=wp.float64)):
#     tid = wp.tid()
#     z[tid] = wp.dot(x[tid], y[tid])

# def dot(x, y, temp):
#     wp.launch(kernel=array_dot, dim=x.shape[0], inputs=[x, y, temp])
#     return array_sum(temp)

@wp.kernel
def step_forward(x0:wp.array(dtype=wp.vec3d), d:wp.array(dtype=wp.vec3d), alpha:wp.float64, x:wp.array(dtype=wp.vec3d)):
    tid = wp.tid()
    x[tid] = x0[tid] + alpha * d[tid]

def conjugate_gradient(A, b, tol=1e-3):
    temp = wp.array(shape=b.shape, dtype=wp.vec3d)
    num_block = b.shape[0]
    diag_inv = wps.bsr_get_diag(A)
    wp.launch(kernel=array_inv, dim=num_block, inputs=[diag_inv], device=b.device)
    x = wp.array(shape=b.shape, dtype=b.dtype)
    x.zero_()
    r = wp.array(shape=b.shape, dtype=b.dtype)
    wp.copy(r, b)
    q = wp.array(shape=b.shape, dtype=b.dtype)
    wp.launch(kernel=array_matmul, dim=num_block, inputs=[diag_inv, r, q], device=b.device)
    
    p = wp.array(shape=b.shape, dtype=b.dtype)
    wp.copy(p, q)

    zTrk = array_inner(r, q)

    residual = zTrk ** 0.5
    if residual < 1e-16:
        return x
    tol = tol * residual
    for iter in range(b.shape[0]):
        residual = zTrk ** 0.5
        if (iter % 50 == 0):
            print(f"[Conjugate Gradient] iter={iter}, residual={residual}")
        if residual < tol:
            break
        wps.bsr_mv(A, p, temp)
        alpha = zTrk / array_inner(temp, p)
        wp.launch(kernel=step_forward, dim=num_block, inputs=[x, p, alpha, x], device=b.device)
        wp.launch(kernel=step_forward, dim=num_block, inputs=[r, temp, -alpha, r], device=b.device)
        wp.launch(kernel=array_matmul, dim=num_block, inputs=[diag_inv, r, q], device=b.device)

        zTrk_last = zTrk
        zTrk = array_inner(q, r)
        beta = zTrk / zTrk_last
        wp.launch(kernel=step_forward, dim=num_block, inputs=[q, p, beta, p], device=b.device)    

    print(f"[Conjugate Gradient] converge iter={iter}, residual={residual}")    
    return x