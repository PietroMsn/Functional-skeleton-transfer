"""Spectral basis, descriptors, functional maps and I/O on synthetic data."""

import numpy as np
import pytest

from skeleton_transfer.config import FunctionalMapConfig
from skeleton_transfer.descriptors import wave_kernel_map, wave_kernel_signature
from skeleton_transfer.functional_map import estimate_functional_map
from skeleton_transfer.shapes import load_mesh, load_target, read_obj
from skeleton_transfer.spectral import compute_laplace_basis

from .conftest import ellipsoid_mesh


# ----------------------------------------------------------------------------
# Spectral basis
# ----------------------------------------------------------------------------


def test_sphere_spectrum():
    vertices, faces = ellipsoid_mesh(40, 80)
    basis = compute_laplace_basis(vertices, faces, n_eigs=16)
    # Unit sphere: eigenvalues l (l + 1) with multiplicity 2 l + 1.
    expected = np.repeat([0, 2, 6, 12], [1, 3, 5, 7])
    np.testing.assert_allclose(basis.evals, expected, atol=0.05 * 12)
    # M-orthonormal eigenfunctions and area of the unit sphere
    gram = basis.evecs.T @ (basis.mass @ basis.evecs)
    np.testing.assert_allclose(gram, np.eye(16), atol=1e-8)
    assert basis.area == pytest.approx(4 * np.pi, rel=0.01)


def test_basis_is_deterministic():
    vertices, faces = ellipsoid_mesh(radii=(1.0, 0.8, 0.6))
    a = compute_laplace_basis(vertices, faces, n_eigs=10)
    b = compute_laplace_basis(vertices, faces, n_eigs=10)
    np.testing.assert_array_equal(a.evecs, b.evecs)


# ----------------------------------------------------------------------------
# Descriptors: the vectorized versions match the original loop implementations
# ----------------------------------------------------------------------------


def _original_wks(evals, evecs, wks_size, variance):
    """``functional_maps.WKS`` from the 2021 code."""
    n = evecs.shape[0]
    evals = evals.copy()
    evals[evals < 1e-6] = 1e-6
    log_E = np.log(evals).T
    e = np.linspace(log_E[1], np.max(log_E) / 1.02, wks_size)
    sigma = (e[1] - e[0]) * variance
    WKS = np.zeros((n, wks_size))
    C = np.zeros((wks_size, 1))
    for i in range(wks_size):
        WKS[:, i] = np.sum(evecs**2 * np.tile(np.exp((-(e[i] - log_E) ** 2) / (2 * sigma**2)), (n, 1)), axis=1)
        C[i] = np.sum(np.exp((-(e[i] - log_E) ** 2) / (2 * sigma**2)))
    return WKS / np.tile(C, (1, n)).T


def _original_wkm(Lambda, Phi, landmarks, wks_size):
    """``goodname.WKM.WKM`` from the 2021 code."""
    n_evecs = Phi.shape[1]
    nv = Phi.shape[0]
    WKM = np.zeros((nv, wks_size * len(landmarks)))
    for li in range(len(landmarks)):
        segment = np.zeros((nv, 1))
        segment[int(landmarks[li])] = 1
        absoluteEigenvalues = np.abs(Lambda)
        emin = np.log(absoluteEigenvalues[1])
        emax = np.log(absoluteEigenvalues[-1])
        s = 7 * (emax - emin) / wks_size
        es = np.linspace(emin + 2 * s, emax - 2 * s, wks_size)
        T = np.exp(
            -np.square(np.tile(np.reshape(np.log(absoluteEigenvalues), (n_evecs, 1)), (1, wks_size)) - np.tile(es.T, (n_evecs, 1)))
            / (2 * np.square(s))
        )
        wkm = np.dot(Phi, T * np.tile(np.dot(Phi.T, segment), (1, T.shape[1])))
        WKM[:, wks_size * li : wks_size * (li + 1)] = wkm
    return WKM


def test_descriptors_match_original():
    vertices, faces = ellipsoid_mesh(radii=(1.0, 0.8, 0.6))
    basis = compute_laplace_basis(vertices, faces, n_eigs=40)
    evals = np.maximum(basis.evals, 1e-6)
    np.testing.assert_allclose(
        wave_kernel_signature(evals, basis.evecs, 50, 7.0), _original_wks(basis.evals, basis.evecs, 50, 7), rtol=1e-10
    )
    landmarks = [0, 100, 500]
    np.testing.assert_allclose(
        wave_kernel_map(evals, basis.evecs, landmarks, 50, 7.0),
        _original_wkm(evals, basis.evecs, landmarks, 50),
        rtol=1e-8,
        atol=1e-12,
    )


# ----------------------------------------------------------------------------
# Functional map
# ----------------------------------------------------------------------------


def test_self_map_is_identity():
    """Mapping a shape onto a remeshed-free copy of itself recovers the identity."""
    vertices, faces = ellipsoid_mesh(radii=(1.0, 0.7, 0.5))
    # Bump to break the ellipsoid's symmetries a little more.
    vertices = vertices + 0.15 * np.exp(-np.sum((vertices - [1, 0, 0]) ** 2, axis=1, keepdims=True) * 4) * vertices
    basis = compute_laplace_basis(vertices, faces, n_eigs=60)
    landmarks = np.array([0, len(vertices) - 1, 300, 600, 900])
    cfg = FunctionalMapConfig(n_eigs_init=10, n_eigs_final=40, zoomout_step=10)
    fmap = estimate_functional_map(basis, basis, landmarks, landmarks, cfg)
    assert np.mean(fmap.p2p == np.arange(len(vertices))) > 0.9


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------


def test_read_obj_quads_and_negative_indices(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1 4//1\nf -4 -3 -2\n")
    vertices, faces = read_obj(str(path))
    assert vertices.shape == (4, 3)
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3], [0, 1, 2]])


def test_load_target_mesh_with_landmarks(tmp_path):
    vertices, faces = ellipsoid_mesh(6, 8)
    mesh = tmp_path / "shape.off"
    lines = ["OFF", f"{len(vertices)} {len(faces)} 0"]
    lines += [" ".join(map(str, v)) for v in vertices] + ["3 " + " ".join(map(str, f)) for f in faces]
    mesh.write_text("\n".join(lines))
    (tmp_path / "lm.txt").write_text("1 2\n3\n")

    target = load_target(str(mesh), landmarks=str(tmp_path / "lm.txt"), one_based_landmarks=True)
    np.testing.assert_array_equal(target.landmarks, [0, 1, 2])
    np.testing.assert_allclose(target.vertices, vertices)
    np.testing.assert_array_equal(load_mesh(str(mesh))[1], faces)
    assert load_target(str(mesh), landmarks=[4, 5], point_cloud=True).faces is None
    with pytest.raises(ValueError):
        load_target(str(mesh))
