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

Adapters:

- `dress4d.py`: 4D-DRESS. Reads the `<subject>_<outfit>.tar.gz` archives as a
  stream (nothing is extracted to disk) or the extracted folders, keeps a few
  frames per sequence (`--stride`, `--frames-per-take`), skips images and
  textures, and computes the exact ground-truth joints from the SMPL parameters
  and the subject gender. Needs `trimesh` (the scan pickles contain trimesh
  arrays) and the official SMPL models (`--smpl-dir`):

  ```bash
  python tools/datasets/dress4d.py path/00123_Outer.tar.gz path/00123_Inner \
      --smpl-dir path/to/SMPL_python_v.1.1.0/smpl/models --out samples/4ddress
  ```

- `manifest.py`: writes `manifest.csv` in a folder of samples, one row per
  sample with its provenance (source archive or folder, subject, outfit,
  sequence, frame, gender, garment label fractions). The exporter rewrites it
  after every run, so the archives can be deleted once a subject is exported.
- `select_frames.py`: evaluation subset with one frame per sequence, chosen by
  farthest-point sampling on the body pose so that the poses are as varied as
  possible; `--keep` preserves the frames of an existing subset:

  ```bash
  python tools/datasets/select_frames.py samples/4ddress --out samples/4ddress_eval \
      --keep samples/4ddress_1pertake
  ```

- `cape.py` (planned): CAPE (per-frame `.npz` with the clothed mesh in SMPL
  topology, pose parameters, per-subject minimally clothed body shape).

Exported samples go to `samples/`, which is ignored by git: the datasets'
licenses do not allow redistributing them.
