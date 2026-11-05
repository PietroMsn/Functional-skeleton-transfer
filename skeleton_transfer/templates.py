"""
Choice of the source among several templates with the same skeleton.

A single template transfers poorly to shapes of very different proportions:
the mean SMAL animal onto a hippopotamus (a long massive trunk on short legs)
gives a functional map that misplaces the whole trunk. When the body model
offers several templates with the same skeleton, landmarks and skinning (the
means of the five SMAL families), the one whose proportions are closest to
the target is chosen before the transfer, without any ground truth:

- :func:`landmark_proportions`: the geodesic distances between the
  landmarks, divided by their mean (invariant to the pose and to the size);
- :func:`proportion_cost`: mean absolute difference of these proportions;
- :func:`select_source`: the source with the lowest cost;
- :func:`smal_family_sources`: the five SMAL family means as sources.
"""

import pickle
from typing import List, Optional, Sequence

import numpy as np

from .descriptors import landmark_geodesics
from .shapes import SMAL_LANDMARKS, SourceShape, TargetShape

#: The SMAL shape clusters, in the order of ``cluster_means`` in ``smal_CVPR2017_data.pkl``.
SMAL_FAMILIES = ["felidae", "canidae", "equidae", "bovidae", "hippopotamidae"]


def landmark_proportions(vertices: np.ndarray, faces: Optional[np.ndarray], landmarks: np.ndarray) -> np.ndarray:
    """Geodesic distances between all pairs of landmarks, divided by their mean.

    Along the mesh edges, or along a k-NN graph for a point cloud
    (``faces is None``, see :func:`.descriptors.landmark_geodesics`).
    """
    d = landmark_geodesics(vertices, faces, landmarks)[np.asarray(landmarks)]
    d = d[np.triu_indices(len(landmarks), 1)]
    return d / d.mean()


def proportion_cost(source: SourceShape, target: TargetShape) -> float:
    """Mean absolute difference of the landmark proportions of a source and a
    target with corresponding landmarks."""
    a = landmark_proportions(source.vertices, source.faces, source.landmarks)
    b = landmark_proportions(target.vertices, target.faces, target.landmarks)
    return float(np.abs(a - b).mean())


def select_source(sources: Sequence[SourceShape], target: TargetShape) -> int:
    """Index of the source whose landmark proportions are closest to the target."""
    return int(np.argmin([proportion_cost(s, target) for s in sources]))


def smal_family_sources(model, data_path: str, landmarks: np.ndarray = SMAL_LANDMARKS) -> List[SourceShape]:
    """The means of the SMAL families (:data:`SMAL_FAMILIES`) as sources.

    ``model`` is the SMAL :class:`~skeleton_transfer.smpl.SMPLModel` and
    ``data_path`` the ``smal_CVPR2017_data.pkl`` file distributed with it.
    """
    with open(data_path, "rb") as f:
        means = pickle.load(f, encoding="latin1")["cluster_means"]
    return [model.as_source(landmarks, np.asarray(m)) for m in means]
