"""
Laplace-Beltrami eigenbasis.

For a shape with ``n`` vertices we compute the generalized eigenproblem

    L phi_i = lambda_i M phi_i

where ``L`` is the (cotangent / point-cloud) Laplacian and ``M`` the lumped
mass matrix, both provided by ``robust_laplacian``. The eigenfunctions are
``M``-orthonormal, so the projection of a function ``f`` (one value per
vertex) onto the first ``k`` eigenfunctions is

    f_hat = Phi[:, :k].T @ M @ f

i.e. ``Phi.T @ M`` is the left pseudo-inverse of ``Phi``.

Coordinates Manifold Harmonics (CMH, Marin et al. 2020) extend the first
``k`` eigenfunctions with 3 functions, the ``M``-orthonormalized residuals of
the x, y, z coordinates: the vertex coordinates are then represented exactly
by the ``k + 3`` functions (:meth:`LaplaceBasis.functions`).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import robust_laplacian
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

logger = logging.getLogger(__name__)

#: Upper bound on the neighbours used to connect a point cloud graph.
MAX_NEIGHBORS = 240


def _n_components(L: sparse.spmatrix) -> int:
    return connected_components(L != 0, directed=False)[0]


@dataclass
class LaplaceBasis:
    """Truncated Laplace-Beltrami eigenbasis of a shape.

    Attributes
    ----------
    evals : (k,) eigenvalues, ascending
    evecs : (n_vertices, k) eigenfunctions, ``M``-orthonormal
    mass : (n_vertices, n_vertices) sparse lumped mass matrix
    vertices : (n_vertices, 3) coordinates of the shape, needed by the CMH
        functions
    """

    evals: np.ndarray
    evecs: np.ndarray
    mass: sparse.spmatrix
    vertices: Optional[np.ndarray] = None

    @property
    def area(self) -> float:
        """Total surface area (sum of the lumped vertex areas)."""
        return float(self.mass.sum())

    def functions(self, k: int, cmh: bool = False) -> np.ndarray:
        """The first ``k`` eigenfunctions, followed by the 3 CMH functions if ``cmh``."""
        phi = self.evecs[:, :k]
        if not cmh:
            return phi
        for residual in (self.vertices - phi @ (phi.T @ (self.mass @ self.vertices))).T:
            f = residual - phi @ (phi.T @ (self.mass @ residual))
            phi = np.column_stack((phi, f / np.sqrt(f @ (self.mass @ f))))
        return phi

    def project(self, f: np.ndarray, k: Optional[int] = None, cmh: bool = False) -> np.ndarray:
        """Spectral coefficients ``Phi.T @ M @ f`` of a vertex function on the
        first ``k`` eigenfunctions (and the CMH functions if ``cmh``).

        ``f`` can be a vector or a matrix (one function per column), e.g. the
        vertex coordinates.
        """
        phi = self.evecs if k is None and not cmh else self.functions(k or self.evecs.shape[1], cmh)
        return phi.T @ (self.mass @ f)


def compute_laplace_basis(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    n_eigs: int,
    is_point_cloud: bool = False,
) -> LaplaceBasis:
    """Compute the first ``n_eigs`` Laplace-Beltrami eigenpairs of a shape.

    Parameters
    ----------
    vertices : (n_vertices, 3) array
    faces : (n_faces, 3) int array, ignored if ``is_point_cloud``
    n_eigs : number of eigenpairs
    is_point_cloud : use the point-cloud Laplacian instead of the mesh one
    """
    if is_point_cloud:
        # A disconnected neighbourhood graph (e.g. the hem of a loose skirt
        # far from the legs) has one zero eigenvalue per component and breaks
        # the functional map: add neighbours until the graph is connected.
        n_neighbors = 30
        L, M = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=n_neighbors)
        while _n_components(L) > 1 and n_neighbors < MAX_NEIGHBORS:
            n_neighbors *= 2
            L, M = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=n_neighbors)
        if n_neighbors > 30:
            logger.warning("Point cloud graph disconnected with 30 neighbours, using %d.", n_neighbors)
    else:
        L, M = robust_laplacian.mesh_laplacian(vertices, np.asarray(faces, dtype=int))
    n_components = _n_components(L)
    if n_components > 1:
        logger.warning(
            "The shape has %d connected components: the functional map is unreliable. "
            "Keep only the largest component or treat the shape as a point cloud.",
            n_components,
        )

    # Shift-invert around 0 to get the smallest eigenvalues. A fixed starting
    # vector makes the result (including the eigenvector signs) reproducible;
    # ARPACK would otherwise start from a random vector.
    v0 = np.ones(vertices.shape[0])
    eig_kwargs = dict(k=n_eigs, M=M, sigma=0.0, which="LM", tol=1e-15, v0=v0)
    try:
        evals, evecs = eigsh(L, **eig_kwargs)
    except RuntimeError:
        # L is singular (constant functions are in its kernel): if the
        # factorization fails, regularize slightly and retry.
        shift = 1e-8 * sparse.identity(vertices.shape[0])
        evals, evecs = eigsh(L - shift, **eig_kwargs)

    return LaplaceBasis(evals=evals, evecs=evecs, mass=M, vertices=np.asarray(vertices, dtype=float))
