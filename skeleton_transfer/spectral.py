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
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import robust_laplacian
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh


@dataclass
class LaplaceBasis:
    """Truncated Laplace-Beltrami eigenbasis of a shape.

    Attributes
    ----------
    evals : (k,) eigenvalues, ascending
    evecs : (n_vertices, k) eigenfunctions, ``M``-orthonormal
    mass : (n_vertices, n_vertices) sparse lumped mass matrix
    """

    evals: np.ndarray
    evecs: np.ndarray
    mass: sparse.spmatrix

    @property
    def area(self) -> float:
        """Total surface area (sum of the lumped vertex areas)."""
        return float(self.mass.sum())

    def project(self, f: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Spectral coefficients ``Phi[:, :k].T @ M @ f`` of a vertex function.

        ``f`` can be a vector or a matrix (one function per column), e.g. the
        vertex coordinates.
        """
        phi = self.evecs if k is None else self.evecs[:, :k]
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
        L, M = robust_laplacian.point_cloud_laplacian(vertices)
    else:
        L, M = robust_laplacian.mesh_laplacian(vertices, np.asarray(faces, dtype=int))

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

    return LaplaceBasis(evals=evals, evecs=evecs, mass=M)
