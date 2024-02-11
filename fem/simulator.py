import warp as wp
import warp.sparse as wps
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from fem.elasticity import compute_elasticity_energy, compute_elasticity_grad, compute_elasticity_hessian
from fem.inertia import compute_x_tilde, compute_inertia_energy, compute_inertia_hessian, compute_inertia_grad
from warp.utils import array_sum, array_inner
from fem.conjugate_gradient import conjugate_gradient, step_forward
from fem.geometry import find_boundary

@wp.struct
class FEMStateStruct:
    X: wp.array(dtype=wp.vec3d)
    x: wp.array(dtype=wp.vec3d)
    v: wp.array(dtype=wp.vec3d)
    x_tilde: wp.array(dtype=wp.vec3d)
    IB : wp.array(dtype=wp.mat33d)
    tet: wp.array(dtype=wp.vec4i)
    surf: wp.array(dtype=wp.vec3i)
    mu: wp.array(dtype=wp.float64)
    lam: wp.array(dtype=wp.float64)
    vol: wp.array(dtype=wp.float64)
    mass: wp.array(dtype=wp.float64)

@wp.kernel
def compute_node_mass(tet_mass: wp.array(dtype=wp.float64), tet: wp.array(dtype=wp.vec4i), node_mass: wp.array(dtype=wp.float64)):
    tid = wp.tid()
    indices = tet[tid]
    m = tet_mass[tid] / wp.float64(4)
    for i in range(4):
        wp.atomic_add(node_mass, indices[i], m)

@wp.kernel
def compute_velocity(xn: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), dt: wp.float64, v: wp.array(dtype=wp.vec3d)):
    tid = wp.tid()
    v[tid] = (x[tid] - xn[tid]) / dt

class Simulator:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.fem_data = FEMStateStruct()
        self.fem_data.x = wp.array([], dtype=wp.vec3d, device=device)
        self.fem_data.v = wp.array([], dtype=wp.vec3d, device=device)
        self.fem_data.x_tilde = wp.array([], dtype=wp.vec3d, device=device)
        self.fem_data.IB = wp.array([], dtype=wp.mat33d, device=device)
        self.fem_data.tet = wp.array([], dtype=wp.vec4i, device=device)
        self.fem_data.surf = wp.array([], dtype=wp.vec3i, device=device)
        self.fem_data.mu = wp.array([], dtype=wp.float64, device=device)
        self.fem_data.lam = wp.array([], dtype=wp.float64, device=device)
        self.fem_data.vol = wp.array([], dtype=wp.float64, device=device)
        self.fem_data.mass = wp.array([], dtype=wp.float64, device=device)
        self.dt = 1e-2
        self.tol = 1e-2

    def add_soft_body(self, x, tet, E=1e5, nu=0.3, rho=1000):
        v0 = x[tet[:, 0]]
        v1 = x[tet[:, 1]]
        v2 = x[tet[:, 2]]
        v3 = x[tet[:, 3]]
        T0 = v1 - v0
        T1 = v2 - v0
        T2 = v3 - v0
        frames = torch.stack([T0, T1, T2], axis=2)
        IB = torch.linalg.inv(frames)
        vol = torch.linalg.det(frames).abs() / 6
        node_mass = torch.zeros(x.shape[0], dtype=torch.float64, device=self.device)
        tet_mass = vol * rho
        wp.launch(kernel=compute_node_mass, dim=tet.shape[0], 
                  inputs=[wp.from_torch(tet_mass, dtype=wp.float64), 
                          wp.from_torch(tet, dtype=wp.vec4i), 
                          wp.from_torch(node_mass, dtype=wp.float64)], device=self.device)
        
        boundary = find_boundary(tet)

        if self.fem_data.x.shape[0] > 0:
            self.fem_data.X = wp.from_torch(torch.cat([wp.to_torch(self.fem_data.X), x], axis=0).contiguous(), dtype=wp.vec3d)
            self.fem_data.x = wp.from_torch(torch.cat([wp.to_torch(self.fem_data.x), x], axis=0).contiguous(), dtype=wp.vec3d)
            self.fem_data.v = wp.from_torch(torch.cat([wp.to_torch(self.fem_data.v), torch.zeros_like(x)], axis=0).contiguous(), dtype=wp.vec3d)
            self.fem_data.tet = wp.from_torch(torch.cat([wp.to_torch(self.fem_data.tet), tet], axis=0).contiguous(), dtype=wp.vec4i)
            self.fem_data.mu = wp.from_torch(torch.cat([wp.to_torch(self.fem_data.mu), E / (2 * (1 + nu)) * torch.ones(tet.shape[0]).to(self.device)], axis=0).contiguous())
            self.fem_data.lam = wp.from_torch(torch.cat([wp.to_torch(self.fem_data.lam), E * nu / ((1 + nu) * (1 - 2 * nu)) * torch.ones(tet.shape[0]).to(self.device)], axis=0).contiguous())
            self.fem_data.vol = wp.from_torch(torch.cat([wp.to_torch(self.fem_data.vol), vol], axis=0).contiguous())
            self.fem_data.mass = wp.from_torch(torch.cat([wp.to_torch(self.fem_data.mass), node_mass], axis=0).contiguous())
            self.fem_data.IB = wp.from_torch(torch.cat([wp.to_torch(self.fem_data.IB), IB], axis=0).contiguous())
            self.fem_data.surf = wp.from_torch(torch.cat([wp.to_torch(self.fem_data.surf), boundary], axis=0).contiguous(), dtype=wp.vec3i)
        else:
            self.fem_data.X = wp.from_torch(x, dtype=wp.vec3d)
            self.fem_data.x = wp.from_torch(x, dtype=wp.vec3d)
            self.fem_data.v = wp.from_torch(torch.zeros_like(x), dtype=wp.vec3d)
            self.fem_data.tet = wp.from_torch(tet, dtype=wp.vec4i)
            self.fem_data.mu = wp.from_torch(E / (2 * (1 + nu)) * torch.ones(tet.shape[0]).to(self.device))
            self.fem_data.lam = wp.from_torch(E * nu / ((1 + nu) * (1 - 2 * nu)) * torch.ones(tet.shape[0]).to(self.device))
            self.fem_data.vol = wp.from_torch(vol)
            self.fem_data.mass = wp.from_torch(node_mass)
            self.fem_data.IB = wp.from_torch(IB.contiguous(), dtype=wp.mat33d)
            self.fem_data.surf = wp.from_torch(boundary, dtype=wp.vec3i)

        self.fem_data.x_tilde = wp.from_torch(torch.zeros([self.fem_data.x.shape[0], 3]).to(self.device), dtype=wp.vec3d)

    def compute_energy(self, x: wp.array(dtype=wp.vec3d)):
        num_x = self.fem_data.x.shape[0]
        num_tet = self.fem_data.tet.shape[0]
        energy = wp.array(shape=(max([num_x, num_tet])), dtype=wp.float64, device=self.device)
        energy.zero_()
        wp.launch(kernel=compute_inertia_energy, dim=num_x, 
                  inputs=[x, self.fem_data.x_tilde, self.fem_data.mass, energy],
                  device=self.device)
        wp.launch(kernel=compute_elasticity_energy, dim=num_tet, 
                  inputs=[x, self.fem_data.tet, self.fem_data.IB, self.fem_data.vol, self.fem_data.mu, self.fem_data.lam, wp.float64(self.dt * self.dt), energy],
                  device=self.device)
        return array_sum(energy)
    
    def compute_grad(self, x: wp.array(dtype=wp.vec3d)):
        num_x = self.fem_data.x.shape[0]
        num_tet = self.fem_data.tet.shape[0]
        grad = wp.array(shape = num_x, dtype=wp.vec3d, device=self.device)
        grad.zero_()
        wp.launch(kernel=compute_inertia_grad, dim=num_x, 
                  inputs=[x, self.fem_data.x_tilde, self.fem_data.mass, grad],
                  device=self.device)
        wp.launch(kernel=compute_elasticity_grad, dim=num_tet, 
                  inputs=[x, self.fem_data.tet, self.fem_data.IB, self.fem_data.vol, self.fem_data.mu, self.fem_data.lam, wp.float64(self.dt * self.dt), grad],
                  device=self.device)
        return grad

    def compute_hess(self, x:wp.array(dtype=wp.vec3d), project_pd: int):
        num_x = self.fem_data.x.shape[0]
        num_tet = self.fem_data.tet.shape[0]
        rows = wp.array(shape=num_x + 16 * num_tet, dtype=wp.int32, device=self.device)
        cols = wp.array(shape=num_x + 16 * num_tet, dtype=wp.int32, device=self.device)
        vals = wp.array(shape=num_x + 16 * num_tet, dtype=wp.mat33d, device=self.device)
        rows.zero_()
        cols.zero_()
        vals.zero_()
        wp.launch(kernel=compute_inertia_hessian, dim=num_x, 
                  inputs=[self.fem_data.mass, 0, rows, cols, vals],
                  device=self.device)
        wp.launch(kernel=compute_elasticity_hessian, dim=num_tet, 
                  inputs=[x, self.fem_data.tet, self.fem_data.IB, self.fem_data.vol, self.fem_data.mu, self.fem_data.lam, num_x, rows, cols, vals, wp.float64(self.dt * self.dt), project_pd],
                  device=self.device)
        bsr_matrix = wps.bsr_zeros(num_x, num_x, block_type=wp.mat33d, device=self.device)
        wps.bsr_set_from_triplets(bsr_matrix, rows, cols, vals)
        return bsr_matrix
    
    def compute_x_tilde(self):
        num_x = self.fem_data.x.shape[0]
        wp.launch(kernel=compute_x_tilde, dim=num_x, inputs=[self.fem_data.x, self.fem_data.v, wp.float64(self.dt), self.fem_data.x_tilde],
                  device=self.device)
    
    def advance_one_step(self):
        self.compute_x_tilde()
        x = wp.array(shape=self.fem_data.x.shape, dtype=wp.vec3d, device=self.device)
        wp.copy(x, self.fem_data.x)
        iter = 0
        while True:
            g = self.compute_grad(x)
            H = self.compute_hess(x, 1)
            d = conjugate_gradient(H, g)
            x0 = wp.array(shape=self.fem_data.x.shape, dtype=wp.vec3d, device=self.device)
            wp.copy(x0, x)
            e0 = self.compute_energy(x)
            alpha = -1.
            wp.launch(kernel=step_forward, inputs=[x0, d, wp.float64(alpha), x], dim=x.shape)
            e1 = self.compute_energy(x)
            while e1 >= e0:
                alpha = alpha / 2
                wp.launch(kernel=step_forward, inputs=[x0, d, wp.float64(alpha), x], dim=x.shape)
                e1 = self.compute_energy(x)
            res = (array_inner(d, d) / (self.fem_data.x.shape[0] * 3)) ** 0.5 / self.dt
            print(f"[Newton] iter={iter}, res={res}")
            iter += 1
            if res < self.tol:
                break
        print(f"[Newton] converge iter={iter}, res={res}")
        wp.launch(kernel=compute_velocity, dim=self.fem_data.x.shape[0], inputs=[self.fem_data.x, x, wp.float64(self.dt), self.fem_data.v], device=self.device)
        wp.copy(self.fem_data.x, x)
        

if __name__ == "__main__":
    device = "cuda:0"
    # device="cpu"
    wp.init()
    torch.set_default_dtype(torch.float64)
    sim = Simulator(device=device)
    from geometry import tet_in_box
    points, elements = tet_in_box([0,0,0], [1,1,1], 1.0, sim.device)
    sim.add_soft_body(points, elements)
    sim.dt = 1e-2
    sim.compute_x_tilde()
    points += 0.2 * torch.rand_like(points)
    delta = torch.rand_like(points)
    delta /= torch.linalg.norm(delta.flatten())
    eps = 1e-4
    P0 = points + eps * delta
    P1 = points - eps * delta
    energy0 = sim.compute_energy(wp.from_torch(P0, dtype=wp.vec3d))
    grad0 = wp.to_torch(sim.compute_grad(wp.from_torch(P0, dtype=wp.vec3d)))
    hess0 = sim.compute_hess(wp.from_torch(P0, dtype=wp.vec3d), 0)
    energy1 = sim.compute_energy(wp.from_torch(P1, dtype=wp.vec3d))
    grad1 = wp.to_torch(sim.compute_grad(wp.from_torch(P1, dtype=wp.vec3d)))
    hess1 = sim.compute_hess(wp.from_torch(P1, dtype=wp.vec3d), 0)
    grad_eps = abs(((energy0 - energy1) / eps - torch.sum(delta * (grad0 + grad1)))).item()
    mul = torch.zeros_like(delta)
    wps.bsr_mv(hess0, wp.from_torch(delta, dtype=wp.vec3d), wp.from_torch(mul, dtype=wp.vec3d), 1, 0)
    wps.bsr_mv(hess1, wp.from_torch(delta, dtype=wp.vec3d), wp.from_torch(mul, dtype=wp.vec3d), 1, 1)
    hess_eps = torch.linalg.norm((grad0 - grad1) / eps - mul).item()

    print(grad_eps, hess_eps)
    
 