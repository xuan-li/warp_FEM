import warp as wp

@wp.kernel
def compute_x_tilde(x: wp.array(dtype=wp.vec3d), 
                    v: wp.array(dtype=wp.vec3d), 
                    dt: wp.float64, 
                    x_tilde: wp.array(dtype=wp.vec3d)):
    tid = wp.tid()
    x_tilde[tid] = x[tid] + dt * v[tid]

@wp.kernel
def compute_inertia_energy(x: wp.array(dtype=wp.vec3d), 
                           x_tilde: wp.array(dtype=wp.vec3d), 
                           mass: wp.array(dtype=wp.float64), 
                           energy: wp.array(dtype=wp.float64)):
    tid = wp.tid()
    diff = x[tid] - x_tilde[tid]
    energy[tid] = energy[tid] + wp.dot(diff, diff) * mass[tid] / wp.float64(2)

@wp.kernel
def compute_inertia_grad(x: wp.array(dtype=wp.vec3d), x_tilde: wp.array(dtype=wp.vec3d), mass: wp.array(dtype=wp.float64), grad: wp.array(dtype=wp.vec3d)):
    tid = wp.tid()
    grad[tid] = grad[tid] + (x[tid] - x_tilde[tid]) * mass[tid]

@wp.kernel
def compute_inertia_hessian(mass: wp.array(dtype=wp.float64), 
                            offset: wp.int32, 
                            rows: wp.array(dtype=wp.int32), 
                            cols:wp.array(dtype=wp.int32), 
                            values: wp.array(dtype=wp.mat33d)):
    tid = wp.tid()
    rows[offset + tid] = tid
    cols[offset + tid] = tid
    values[offset + tid] = wp.identity(n=3, dtype=wp.float64) * mass[tid]