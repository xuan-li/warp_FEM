import torch
import warp as wp

@wp.kernel
def fill_grid(nodes: wp.array(dtype=wp.vec3d), nx: wp.int32, ny: wp.int32, nz: wp.int32):
    ix, iy, iz = wp.tid()
    index = ix * ny * nz + iy * nz + iz
    nodes[index] = wp.vec3d(wp.float64(ix), wp.float64(iy), wp.float64(iz))

class vec8i(wp.types.vector(length=8, dtype=wp.int32)):
    pass

@wp.kernel
def fill_element(elements: wp.array(dtype=wp.vec4i), nx: wp.int32, ny: wp.int32, nz: wp.int32):
    ix, iy, iz = wp.tid()
    index = ix * ny * nz + iy * nz + iz
    vertices = vec8i(ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz,
                ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz + 1,
                ix * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + iz,
                ix * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + iz + 1,
                (ix + 1) * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz,
                (ix + 1) * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz + 1,
                (ix + 1) * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + iz,
                (ix + 1) * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + iz + 1)
    if ((not(ix % 2)) and (not(iy % 2)) and (not(iz % 2))) or ((ix % 2) and (iy % 2) and (iz % 2)):
        elements[index * 6] = wp.vec4i(vertices[0], vertices[4], vertices[2], vertices[5])
        elements[index * 6 + 1] = wp.vec4i(vertices[0], vertices[2], vertices[1], vertices[5])
        elements[index * 6 + 2] = wp.vec4i(vertices[1], vertices[2], vertices[3], vertices[5])
        elements[index * 6 + 3] = wp.vec4i(vertices[2], vertices[5], vertices[4], vertices[6])
        elements[index * 6 + 4] = wp.vec4i(vertices[2], vertices[7], vertices[5], vertices[6])
        elements[index * 6 + 5] = wp.vec4i(vertices[2], vertices[3], vertices[5], vertices[7])
    elif ((ix % 2) and (not(iy % 2)) and (not(iz % 2))) or ((not(ix % 2)) and (iy % 2) and (iz % 2)):
        # x reflect
        elements[index * 6] = wp.vec4i(vertices[0], vertices[4], vertices[6], vertices[1])
        elements[index * 6 + 1] = wp.vec4i(vertices[6], vertices[4], vertices[5], vertices[1])
        elements[index * 6 + 2] = wp.vec4i(vertices[6], vertices[5], vertices[7], vertices[1])
        elements[index * 6 + 3] = wp.vec4i(vertices[1], vertices[6], vertices[0], vertices[2])
        elements[index * 6 + 4] = wp.vec4i(vertices[3], vertices[6], vertices[1], vertices[2])
        elements[index * 6 + 5] = wp.vec4i(vertices[7], vertices[6], vertices[1], vertices[3])
    elif ((not(ix % 2)) and (iy % 2) and (not(iz % 2))) or ((ix % 2) and (not(iy % 2)) and (iz % 2)):
        # y reflect
        elements[index * 6] = wp.vec4i(vertices[6], vertices[2], vertices[0], vertices[7])
        elements[index * 6 + 1] = wp.vec4i(vertices[0], vertices[2], vertices[3], vertices[7])
        elements[index * 6 + 2] = wp.vec4i(vertices[0], vertices[3], vertices[1], vertices[7])
        elements[index * 6 + 3] = wp.vec4i(vertices[7], vertices[0], vertices[6], vertices[4])
        elements[index * 6 + 4] = wp.vec4i(vertices[5], vertices[0], vertices[7], vertices[4])
        elements[index * 6 + 5] = wp.vec4i(vertices[1], vertices[0], vertices[7], vertices[5])
    elif ((not(ix % 2)) and (not(iy % 2)) and (iz % 2)) or ((ix % 2) and (iy % 2) and (not(iz % 2))):
        # z reflect
        elements[index * 6] = wp.vec4i(vertices[5], vertices[1], vertices[3], vertices[4])
        elements[index * 6 + 1] = wp.vec4i(vertices[3], vertices[1], vertices[0], vertices[4])
        elements[index * 6 + 2] = wp.vec4i(vertices[3], vertices[0], vertices[2], vertices[4])
        elements[index * 6 + 3] = wp.vec4i(vertices[4], vertices[3], vertices[5], vertices[7])
        elements[index * 6 + 4] = wp.vec4i(vertices[6], vertices[3], vertices[4], vertices[7])
        elements[index * 6 + 5] = wp.vec4i(vertices[2], vertices[3], vertices[4], vertices[6])

def tet_in_box(min_corner, max_corner, dx, device):
    min_corner = torch.tensor(min_corner, dtype=float).to(device)
    max_corner = torch.tensor(max_corner, dtype=float).to(device)
    box_size = max_corner - min_corner
    resolution = (box_size / dx).round().to(torch.int32)
    nx = resolution[0]
    ny = resolution[1]
    nz = resolution[2]
    points = torch.zeros((nx + 1) * (ny + 1) * (nz + 1), 3).to(device)
    wp.launch(kernel=fill_grid, dim=(nx+1, ny+1, nz+1), inputs=[wp.from_torch(points, dtype=wp.vec3d), nx+1, ny+1, nz+1], device=device)
    points = (points * dx) + min_corner[None]
    
    elements = torch.zeros(nx * ny * nz * 6, 4).to(torch.int32).to(device)
    wp.launch(kernel=fill_element, dim=(nx, ny, nz), inputs=[wp.from_torch(elements, dtype=wp.vec4i), nx, ny, nz], device=device)
    
    return points, elements

def find_boundary(tet):
    element_boundary = torch.cat([tet[:, [2, 1, 0], None], tet[:, [0, 1, 3], None], tet[:, [1, 2, 3], None], tet[:, [2, 0, 3], None]], axis=1)
    element_boundary = element_boundary.reshape([-1, 3]).cpu().numpy()
    half_elements = {}
    for btri in element_boundary:
        key = tuple(sorted(btri.tolist()))
        if key not in half_elements:
            half_elements[key] = []
        half_elements[key].append(btri.tolist())
    boundary_tris = []
    for e in half_elements.values():
        if len(e) == 1:
            boundary_tris.append(e[0])
    return torch.tensor(boundary_tris, dtype=torch.int32)