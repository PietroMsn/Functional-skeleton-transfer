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

## Reproducing the numbers

- Parity and regressor: `pytest tests/test_parity.py` (needs `data/`), and
  `tools/make_original_reference.py` for the original outputs.
- Pose invariance (Tab. 1): `test_regressor_is_pose_invariant` in
  `tests/test_parity.py` (the table uses 20 poses per sigma).
- Variants, diagnostics and convergence (Tab. 3-4): `tools/benchmark_synthetic.py -n 6` at commit
  `20fc426` (offline analysis of the saved stage histories for the stopping rule).
- Synthetic benchmark (Tab. 2): `tools/benchmark_synthetic.py -n 8` at commit
  `77fccbb`; prior weights evaluated offline from the saved joints.
