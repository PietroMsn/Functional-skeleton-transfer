import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Folder with the example SMPL / DFAUST files (not shipped with the package).
DATA_DIR = os.environ.get("SKELETON_TRANSFER_DATA", os.path.join(REPO_ROOT, "data"))
REFERENCE = os.path.join(REPO_ROOT, "tests", "data", "original_reference.npz")


def ellipsoid_mesh(n_lat=24, n_lon=48, radii=(1.0, 1.0, 1.0)):
    """Closed triangle mesh of an ellipsoid (UV sphere with two poles)."""
    theta = np.linspace(0, np.pi, n_lat + 1)[1:-1]
    phi = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    ring = np.stack((np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)), axis=-1).reshape(-1, 3)
    vertices = np.vstack(([0, 0, 1], ring, [0, 0, -1])) * np.asarray(radii)

    def v(i, j):  # ring i, longitude j -> vertex index
        return 1 + i * n_lon + (j % n_lon)

    south = len(vertices) - 1
    faces = []
    for j in range(n_lon):
        faces.append([0, v(0, j), v(0, j + 1)])
        faces.append([south, v(n_lat - 2, j + 1), v(n_lat - 2, j)])
        for i in range(n_lat - 2):
            faces.append([v(i, j), v(i + 1, j), v(i + 1, j + 1)])
            faces.append([v(i, j), v(i + 1, j + 1), v(i, j + 1)])
    return vertices, np.array(faces)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
