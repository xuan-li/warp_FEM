import sys
sys.path.append(".")

import warp as wp
from fem.simulator import Simulator
import torch
from fem.geometry import tet_in_box
from fem.io import write_ply

torch.set_default_dtype(torch.float64)

wp.init()

sim = Simulator()

points, elements = tet_in_box([0,0,0], [1,1,1], 0.1, sim.device)
sim.add_soft_body(points, elements, 1e6, 0.3, 1e3)
segment_x = torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]], dtype=torch.float64, device=sim.device)
segment_edge = torch.tensor([[0, 1]], dtype=torch.int32, device=sim.device)

sim.dt = 1e-2
sim.tol = 1e-2

@wp.kernel
def enforce_BC(x: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d)):
    tid = wp.tid()
    if x[tid][0] <= wp.float64(0.00001):
        v[tid] = wp.vec3d(wp.float64(-100),wp.float64(0),wp.float64(0))
    elif x[tid][0] >= wp.float64(1. - 0.00001):
        v[tid] = wp.vec3d(wp.float64(100),wp.float64(0),wp.float64(0))

import os
output_dir = "output/simple_elasticity"
os.makedirs(output_dir, exist_ok=True)
write_ply(os.path.join(output_dir, f"{0}.ply"), sim.fem_data.x.numpy(), sim.fem_data.surf.numpy())
for f in range(100):
    if f < 50:
        wp.launch(enforce_BC, sim.fem_data.v.shape, inputs=[sim.fem_data.x, sim.fem_data.v])
    sim.advance_one_step()
    write_ply(os.path.join(output_dir, f"{f+1}.ply"), sim.fem_data.x.numpy(), sim.fem_data.surf.numpy())
