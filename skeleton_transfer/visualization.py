"""
Interactive visualization with PyVista (optional dependency).

:func:`plot_transfer` shows three side-by-side views:

1. the source shape with its original skeleton;
2. the target with the joints transferred through the functional map;
3. the target with the joints transferred through the point-to-point map.
"""

from typing import Optional

import numpy as np

from .shapes import SourceShape, TargetShape
from .transfer import TransferResult


def faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
    """Convert an ``(n_faces, 3)`` triangle array to PyVista's flat format.

    PyVista expects ``[3, i0, i1, i2, 3, j0, j1, j2, ...]``: each polygon is
    prefixed by its number of vertices.
    """
    faces = np.asarray(faces, dtype=int)
    counts = np.full((len(faces), 1), faces.shape[1], dtype=int)
    return np.hstack((counts, faces)).ravel()


def _add_shape(plotter, vertices: np.ndarray, faces: Optional[np.ndarray], opacity: float = 0.35):
    import pyvista as pv

    if faces is None:
        plotter.add_mesh(pv.PolyData(vertices), opacity=opacity, point_size=2)
    else:
        plotter.add_mesh(pv.PolyData(vertices, faces_to_pyvista(faces)), opacity=opacity)


def _add_skeleton(
    plotter,
    joints: np.ndarray,
    parents: np.ndarray,
    joint_color: str = "red",
    bone_color: str = "blue",
    highlight_joint: Optional[int] = None,
):
    """Draw the joints as points and each bone as a segment joint -> parent."""
    import pyvista as pv

    plotter.add_mesh(pv.PolyData(joints), color=joint_color, point_size=10, render_points_as_spheres=True)
    for j, parent in enumerate(parents):
        if j == parent:  # root
            continue
        plotter.add_mesh(pv.Line(joints[parent], joints[j]), color=bone_color, line_width=3)
    if highlight_joint is not None:
        plotter.add_mesh(
            pv.PolyData(joints[highlight_joint][None, :]),
            color="green",
            point_size=14,
            render_points_as_spheres=True,
        )


def plot_transfer(
    source: SourceShape,
    target: TargetShape,
    result: TransferResult,
    highlight_joint: Optional[int] = None,
    screenshot: Optional[str] = None,
) -> None:
    """Show the source skeleton and the two transferred skeletons.

    Parameters
    ----------
    source, target : the shapes passed to ``transfer_skeleton``
    result : output of ``transfer_skeleton``
    highlight_joint : optional joint index to draw in green on the targets
    screenshot : if given, save the window to this image file instead of
        opening an interactive window
    """
    import pyvista as pv

    plotter = pv.Plotter(shape=(1, 3), border=False, window_size=(1800, 700), off_screen=screenshot is not None)

    plotter.subplot(0, 0)
    _add_shape(plotter, source.vertices, source.faces)
    _add_skeleton(plotter, source.joints, source.parents)
    plotter.add_text("Source skeleton", font_size=10)

    plotter.subplot(0, 1)
    _add_shape(plotter, target.vertices, target.faces)
    _add_skeleton(plotter, result.joints_spectral, source.parents, highlight_joint=highlight_joint)
    plotter.add_text("Spectral transfer (functional map)", font_size=10)

    plotter.subplot(0, 2)
    _add_shape(plotter, target.vertices, target.faces)
    _add_skeleton(plotter, result.joints_pointwise, source.parents, highlight_joint=highlight_joint)
    plotter.add_text("Pointwise transfer (p2p map)", font_size=10)

    if screenshot is not None:
        plotter.show(screenshot=screenshot)
    else:
        plotter.show()
