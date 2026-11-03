# Paper draft: A Functional Skeleton Transfer (extension)

LaTeX draft of the *Method* and *Experiments* sections of an extension of
Musoni, Marin, Melzi, Castellani - *A Functional Skeleton Transfer*, PACMCGIT 2021.

This branch (`paper/draft`) only contains the paper. The code lives in the
other branches of the repository; every section and number refers to the code
commit noted next to it in `main.tex`. Only changes whose effect has been
verified are included.

Build:

```bash
latexmk -pdf main.tex
```

## Changelog

| date | code commit | content |
| --- | --- | --- |
| 2026-10-26 | `4276de0` (refactor/package-structure) | closed-form regressor and initial functional map, merged locality/sparsity terms, parity with the 2021 code, pose-invariance experiment |
| 2026-10-26 | `77fccbb` (feature/target-skeleton-priors) | point-cloud connectivity, skeleton priors, plausibility check, synthetic benchmark with clothing |
| 2026-10-26 | `20fc426` (feature/target-skeleton-priors) | neutral SMPL source; robust correspondence (limb landmarks, restarts, skeleton-guided refinement), unsupervised diagnostics, convergence and stopping criterion |
| 2026-10-27 | after `27c1ce8` (feature/target-skeleton-priors, stopping rule not yet committed) | stopping rule in the method; real scans: 4D-DRESS, one frame per sequence (Tab. 5) |
| 2026-10-27 | after `2daabbb` (not yet committed) | restart selection after the refinement (segment consistency), evaluated on the 13 targets where the restarts trigger |
| 2026-10-28 | `f0d7efc` + `--shard` (feature/target-skeleton-priors) | real scans extended to 88 frames of ten subjects with varied poses (Tab. 5), refined restarts validated on the 63 new scans |
| 2026-10-28 | `05187bc` (feature/target-skeleton-priors) | comparison with an optimization-based SMPL fitting |
| 2026-10-28 | `05187bc` (feature/target-skeleton-priors) | real scans extended to 185 frames of all nineteen downloaded subjects (Tab. 5, the 88 previous frames kept), statistics and SMPL fitting comparison updated |
| 2026-10-31 | `27a49c0` (feature/target-skeleton-priors) | bijective ZoomOut in the method and as default; plain vs bijective on the synthetic benchmark (new table); real scans re-run with the bijective ZoomOut (Tab. 6, plain ZoomOut kept as last row), statistics and SMPL fitting comparison updated; GPU nearest-neighbour search |
| 2026-10-31 | `d3f1581` (feature/target-skeleton-priors) | per-chain restarts in the method; analysis of the restart choice (whole-candidate scores, fusions, per-chain oracle) and per-chain variant on the real scans (Tab. 6) and the synthetic benchmark; SMPL fitting comparison updated |
| 2026-11-01 | `cd70d3d` (feature/target-skeleton-priors) | loose clothing: curvature-based detection of covered source vertices with a metric neighbourhood, visible-only regressor; oracles, failed partial map, seed-0/seed-1 synthetic table, real scans (Tab. 6) |
| 2026-11-03 | after `cd70d3d` (feature/target-skeleton-priors, not yet committed) | landmark Voronoi cells after ZoomOut (default) and geodesic descriptors in the method; unified benchmark (dev split, 890 samples: 4D-DRESS, CloSe-Di, BUFF, synthetic adults / clothed / children / SMAL animals) with the loose clothing and Voronoi variants (new table); study of the functional map ingredients on 57 samples (new table); loose clothing on by default |

## Reproducing the numbers

- Unified benchmark (dev split): `tools/benchmark_unified.py DATASET_ROOT --split dev --smal
  smal_CVPR2017.pkl --shard I/8 --out unified_sI.pkl` (variants: baseline, per-chain restarts,
  loose clothing, Voronoi; `--variants "... loose clothing, geodesic + Voronoi"` for the last
  column), merged with `--report`. The dataset and the provenance of every sample (`manifest.csv`)
  come from the `unified-body-dataset` project. The map study (57 samples) is a scratch script,
  not in the repository.

- Loose clothing (synthetic table and last real row before the plain ZoomOut):
  `tools/benchmark_synthetic.py -n 6 [--seed 1]` and `tools/evaluate_scans.py --shard i/7`
  with the variant "all + priors, per-chain restarts, loose clothing".

- Parity and regressor: `pytest tests/test_parity.py` (needs `data/`), and
  `tools/make_original_reference.py` for the original outputs.
- Pose invariance (Tab. 1): `test_regressor_is_pose_invariant` in
  `tests/test_parity.py` (the table uses 20 poses per sigma).
- Variants, diagnostics and convergence (Tab. 3-4): `tools/benchmark_synthetic.py -n 6` at commit
  `20fc426` (offline analysis of the saved stage histories for the stopping rule).
- Real scans (Tab. 5): `tools/datasets/dress4d.py` on all 28 archives (subjects
  00123, 00127, 00129, 00134, 00135, 00137, 00140, 00147, 00149, 00151, 00160,
  00167, 00174, 00179, 00180, 00185, 00188, 00190, 00191; stride 20), then
  `tools/datasets/select_frames.py samples/4ddress --out samples/4ddress_eval_full
  --keep samples/4ddress_eval` (one frame per sequence, the 88 frames of the
  previous version kept; the provenance of every scan is in `manifest.csv`) and
  `tools/evaluate_scans.py <folder> --smpl-dir <SMPL v1.1.0 models> --shard I/N`,
  merged with `--report`; SMPL fitting with `tools/fit_smpl_baseline.py` and
  `--compare`. Per-subject, plausibility and restart statistics:
  `tools/analyze_scans.py <eval pickles> --samples samples/4ddress --subset
  samples/4ddress_eval` (commit `fa403c7`). With the bijective ZoomOut (default
  since `27a49c0`): the same commands write `samples/eval_bijdef_s*.pkl`; the
  plain ZoomOut is the variant suffix `, plain ZoomOut` of `--variants`.
- Per-chain restarts (after `27a49c0`): the variant `all + priors, per-chain restarts` added with
  `--resume` to copies of the bijective runs (`samples/eval_chains_s*.pkl`,
  `samples/bench_chains_*.pkl`); SMPL comparison with `tools/fit_smpl_baseline.py --compare
  samples/fit_smpl_s*.pkl samples/fit_smpl_new.pkl --with samples/eval_chains_s*.pkl --reference
  "all + priors, per-chain restarts"`.
- Plain vs bijective ZoomOut on the synthetic benchmark: `tools/benchmark_synthetic.py
  -n 6 --conditions <c>` per condition (seed 0), with `--plain-zoomout` for the
  plain variants.
- Synthetic benchmark (Tab. 2): `tools/benchmark_synthetic.py -n 8` at commit
  `77fccbb`; prior weights evaluated offline from the saved joints.
