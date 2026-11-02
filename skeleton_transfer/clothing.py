"""
Loose clothing: source vertices whose body part is hidden under a garment.

A skirt or a dress replaces the two legs with one tube: the source vertices of
the thighs and knees have no counterpart on the target, and the point-to-point
map sends them onto the garment, far from the hidden legs. The joints
regressed from those points are wrong (the knees are pulled onto the skirt),
and the functional map fitted on them drags the hips down.

The garment is detected without the skeleton (which is itself wrong under
the garment): a garment is much flatter than the body part it covers, so the
local radius of curvature at the matched target point is much larger than at
the source vertex. The log-ratio of the two radii, smoothed over the source
surface (the covered parts are large connected regions), is thresholded.
The covered vertices are then left out of the joint regressor
(:func:`visible_regressor_mask`). (Leaving them out of the functional map
fits too, i.e. a partial map, leaves the map unconstrained on the garment
and diverges.)
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def local_radius(points: np.ndarray, neighbourhood: float, min_neighbors: int = 8) -> np.ndarray:
    """Radius of curvature of the surface at every point, ``(n_points,)``.

    The normals are estimated by PCA of the points within distance
    ``neighbourhood`` (at least the ``min_neighbors`` nearest ones); the
    radius is the median, over the same neighbours, of the distance divided by
    the angle between the two normals (for a sphere of radius ``r`` this is
    ``r``; for a flat region it is large). The neighbourhood is metric, not a
    number of neighbours, so that the radius is measured at the same scale on
    a mesh refined on the hands and face (SMPL) and on a uniformly sampled
    scan; with ``k`` nearest neighbours the fingers and the nose of the mesh
    look much more curved than the smooth hands and head of a scan.
    """
    tree = cKDTree(points)
    nearest = tree.query(points, min_neighbors)[1]
    neighbours = [
        np.array(nb) if len(nb) >= min_neighbors else nearest[i]
        for i, nb in enumerate(tree.query_ball_point(points, neighbourhood))
    ]
    normals = np.empty_like(points)
    for i, nb in enumerate(neighbours):
        local = points[nb] - points[nb].mean(axis=0)
        normals[i] = np.linalg.eigh(local.T @ local)[1][:, 0]
    radius = np.empty(len(points))
    for i, nb in enumerate(neighbours):
        nb = nb[nb != i]
        dist = np.linalg.norm(points[nb] - points[i], axis=1)
        angle = np.arccos(np.clip(np.abs(normals[nb] @ normals[i]), 0, 1)) + 1e-3
        radius[i] = np.median(dist / angle)
    return radius


def adjacency(vertices: np.ndarray, faces=None, n_neighbors: int = 8) -> sp.csr_matrix:
    """Symmetric 0/1 adjacency of the mesh edges, or of a k-NN graph without faces."""
    n = len(vertices)
    if faces is not None:
        rows, cols = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    else:
        idx = cKDTree(vertices).query(vertices, n_neighbors + 1)[1][:, 1:]
        rows, cols = np.repeat(np.arange(n), n_neighbors), idx.ravel()
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), (n, n)).tocsr()
    return ((graph + graph.T) > 0).astype(float)


def smooth(values: np.ndarray, graph: sp.csr_matrix, n_iterations: int) -> np.ndarray:
    """``n_iterations`` steps of averaging every value with its neighbours."""
    degree = np.asarray(graph.sum(axis=1)).ravel() + 1
    for _ in range(n_iterations):
        values = (graph @ values + values) / degree
    return values


def covered_vertices(
    source_radius: np.ndarray,
    target_radius: np.ndarray,
    p2p: np.ndarray,
    scale: float,
    graph: sp.csr_matrix,
    threshold: float,
    n_smooth: int,
) -> np.ndarray:
    """Boolean mask of the source vertices matched onto a loose garment.

    ``scale`` is the target / source size ratio; ``threshold`` applies to the
    smoothed ``log(target radius / (scale * source radius))``.
    """
    score = np.log(target_radius[p2p] / (scale * source_radius))
    return smooth(score, graph, n_smooth) > threshold


def visible_regressor_mask(
    mask: np.ndarray, visible: np.ndarray, skinning_weights: np.ndarray, parents: np.ndarray, threshold: float
) -> np.ndarray:
    """Locality mask of the joint regressor restricted to the visible vertices.

    A joint whose mask is (almost) entirely covered falls back to the visible
    vertices skinned to its parent, its own or a child bone, so that it is
    extrapolated from the visible part of the limb (e.g. a knee under a skirt
    from the shin).
    """
    out = mask * visible[None]
    for j in range(len(parents)):
        if out[j].sum() < 10:
            bones = [j, parents[j]] + [c for c in range(len(parents)) if parents[c] == j and c != j]
            out[j] = (skinning_weights[:, bones].sum(axis=1) > threshold) & visible
    return out
