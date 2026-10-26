# Dataset adapters

Each adapter converts the frames of a dataset into the common sample format of
`skeleton_transfer.scans` (`ScanSample`, saved with `save_sample`), which
`tools/evaluate_scans.py` consumes. An adapter must provide, per frame:

- `scan_vertices`, `scan_faces` (or `None` for point clouds): the scan;
- `smpl_vertices`: the 6890 registered SMPL vertices, in the same frame as the scan;
- `gender` (`male` / `female` / `neutral`) and a `category` for the reports
  (e.g. the clothing type);
- optionally `joints` (forward-kinematics joints, if the dataset gives pose and
  shape) and `scan_labels` (per-vertex semantic labels).

Planned adapters (to be written against the downloaded data, so that the
field names are checked on the real files):

- `cape.py`: CAPE (per-frame `.npz` with the clothed mesh in SMPL topology,
  pose parameters, per-subject minimally clothed body shape);
- `dress4d.py`: 4D-DRESS (`Meshes_pkl/mesh-f*.pkl` scans, `SMPL/mesh-f*_smpl.ply`
  registrations, `Semantic/labels`, `basic_info.pkl` rotation).
