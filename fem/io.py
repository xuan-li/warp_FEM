from plyfile import PlyData, PlyElement
import numpy as np

def write_ply(filename, points, tris):
    vertex = np.array([tuple(v) for v in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    face_data = tris.astype('i4')
    face = np.empty(len(face_data),dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = face_data
    el = PlyElement.describe(vertex, 'vertex')
    el2 = PlyElement.describe(face, 'face')
    PlyData([el, el2]).write(filename)