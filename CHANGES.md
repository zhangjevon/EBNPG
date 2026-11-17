# Changes relative to the MPG repository

This release is a modified version of <https://github.com/lfprudente/MPG>
(GNU GPL-3.0). The files below are grouped by their relation to that
repository. Files marked *modified* differ from the corresponding MPG file;
files marked *added* are new in this release.

## Revision 3.2 --- tag `v3.2-jogo-r3`

No number printed in the manuscript changes. One count changes, and the way it is obtained.

At 128 of the returned points of the common-stationarity study, neither the first pass nor the re-solve
ladder certified a minimiser of the proximal subproblem, and the per-point record was taken at the point
the solver had returned. The manuscript states the two criteria at the minimiser of that subproblem, so
the displacement of the returned point is not the quantity it defines, and the two differ by more than
the tolerance at two of the 128: MGH16 with CNPG-Max at run 63 (returned point 7.56e-4 from `x`,
minimiser 8.29e-5) and MOP5 with EBNPG-N0 at run 7 (returned 1.15e-4, minimiser 9.2e-15). With the
criterion read at the minimiser, the displacement column passes 31 points rather than 29 --- the same 31
that meet the criticality criterion --- and 103 of the 134 uncertified points pass neither, the six at
which the subproblem has no solution included.

Added (`src/r3_postprocessing/`):

- `certify_adjudicated.m` --- rebuilds each of the 128 subproblems at the stored final iterate and writes
  it to a `.mat` file in the variables of the proximal subproblem: `p = x + d`, the simplex variables of
  the `l1` epigraph of the constraint functions, and the free scaling variable that makes the epigraph
  exact. The subproblem is handed over in a form a solver of the caller's choice can take, so the
  verdict does not have to come from the code whose output was in doubt.
- `certify_adjudicated.py` --- solves the exported subproblems with OSQP and rewrites
  `adjudicated.csv`. OSQP is used deliberately: the disputed points came out of `quadprog`'s
  interior-point method, and an independent solver is what makes the second reading independent. A
  candidate is accepted only when the `p`-block of the KKT residual of the model, which at an optimum is
  an element of the subdifferential of the objective, bounds its distance to the minimiser; the radius
  is written to the record, and a verdict is entered only when the whole interval it leaves for the
  value and for the displacement lies on one side of the tolerance.

Changed (`Code_Data/crit_out_r4/`):

- `adjudicated.csv` now carries the certified value and displacement of the minimiser (`theta_star`,
  `dnorm_star`), the radius that certifies them, the solver status and the two verdicts
  (`pass_theta`, `pass_dnorm`); the readings at the returned point are kept beside them as `ret_theta`
  and `ret_dnorm`, and the file as it stood in `v3.1` is preserved as `adjudicated_uncertified.csv`.
  Over the 128 points the radius has median 1.0e-7 and maximum 2.2e-5, and is at most 1.25e-6 at the 31
  points that pass.
- `common_sr.json` regenerated from the new record. The two averages the change touches move by 0.024
  percentage points (CNPG-Max 92.190 -> 92.214 in the displacement column, EBNPG-N0 90.262 -> 90.286),
  which is invisible at the precision the table is printed to, and the rank correlation between the
  displacement column and the flag column becomes 0.998 rather than 1.000: CNPG-Max and EBNPG-N5, tied
  at 92.1% under the flag, separate in the third decimal (92.214% and 92.190%).

Unchanged: all 72 cells of Table 4 (`check_table4.py`, 72 of 72), Figures 1 and 2, every other table, and
every other JSON.

## Revision 3.1 --- tag `v3.1-jogo-r3`

No reported number changes with respect to `v3.0-jogo-r3`. The differences are in how the package
runs and in how its files account for themselves; both were audit findings on `v3.0`, where several
scripts ran only from one particular working directory and one summary file no longer reproduced.

Changed:

- Every entry point and post-processing script now resolves its inputs from its own location, by
  searching upwards for the folder that contains `Code/` and `Code_Data/`, instead of assuming a
  fixed directory depth and a working directory of `Code_Data/`. `recompute_common_stationarity_r4.m`
  also creates its output folder *after* changing into the data directory, which is the point at
  which the previous version failed on a bare output name.
- `recompute_sr.py` now applies the same certification mask and the same verdict-level adjudication
  as `src/r3_postprocessing/r4_table4.py`, and reproduces all 72 cells of Table 4. In `v3.0` it
  omitted the adjudication and returned 95.1 for two rows in which the manuscript prints 95.2.
- `cmh_test.json`, `paired_test.json`: regenerated from the shipped `results/`. The `v3.0` copies had
  been produced from an earlier `results/` and no longer reproduced (65 of 88 and 49 of 66 numeric
  keys differ; the mean difference for MPG-Explicit is 4.524 percentage points, not 4.762).
- `r2_oracle_cost.py` writes `per_iter_g` and `per_iter_h` under separate keys, and runs with no
  arguments (defaults: `results/`, `oracle_cost_r2.json`). `v3.0` packed the pair into a single key
  named `per_iter_h`, which invites the reader to take the first entry as `n_h`.
- `mcnemar.py` is labelled for what it is: the superseded instance-level analysis, retained because
  its output documents why the pairing was taken at problem level. `run_problem.m` redraws the
  starting point per algorithm slot, so the 4,200 instances of a problem are not paired across
  algorithms; the manuscript states this and reports `cmh_test.py` and `paired_test.py` instead.
  The `v3.0` README described `mcnemar.py` as the per-problem McNemar test of the fixed-kernel study,
  which it is not.
- `verify_manuscript.py` derives the cited release tag, commit and DOI from the manuscript and checks
  that they agree with each other, instead of asserting the `v3.0` strings, which made the checker
  fail as soon as the package was re-released.
- Every JSON written by these scripts now carries a `provenance` block naming the script, the inputs,
  the success flag and the table it feeds, and marking the files that enter no reported number.
- `build_kernel.py` reads `ne_full/` by an absolute path derived from the script, and writes
  `kernel_study.json` as `{"provenance": ..., "rows": [...]}`; `build_tab5.py` and the two
  `oracle_cost*.py` gained the same block.
- Licence headers: previously inserted after the first line, which duplicated the
  `# -*- coding: utf-8 -*-` line of the Python files; they are now on top. Files derived from the MPG
  repository keep their own `Modified from the MPG repository` header.

Added:

- `src/check_table4.py` --- parses the LaTeX source of Table 4 and compares all 72 cells with
  `common_sr.json`, so that the table and the data cannot drift apart unnoticed.

Removed:

- `src/r2_perf_profiles.py` --- superseded by `src/generate_perf_profiles.py`, which reproduces the
  printed Figure 2 pixel for pixel.

## Revision 3 --- tag `v3.0-jogo-r3`

The sources in `src/` have been regenerated from the tree that produced the numbers of the third
revision; apart from the items below, the differences are whitespace and licence-header
normalisation.

Changed:

- `aggregate_all.py` --- the stationarity table it reads is now the certified `crit_out_r4/`
  (`EBPGM_CRIT=crit_out` restores the previous behaviour). With the old default the three Table 4
  columns came out without the certification mask.
- `generate_pareto_plots.py` --- Figure 1 is now drawn in the plane of the composite objectives
  `F_i = f_i + g_i` rather than of their smooth parts, and the six problems it selects are the ones
  named in Subsection 6.8 of the manuscript (it previously fell back, silently, to the first six names
  in alphabetical order). Its data directory now resolves to `Code_Data/` instead of a directory that
  does not exist.
- `generate_perf_profiles.py` --- added (Figure 2); `r2_perf_profiles.py` is the earlier version.
- `export_all_results.m`, `export_archive.m`, `noneuclid_run.m`, `qp_count_full.m`, `qp_count_study.m`,
  `qp_one.m`, `recompute_sr.py`, `build_kernel.py`, `build_tab5.py`, `regen_summary.py`, `cmh_test.py`,
  `paired_test.py`, `mcnemar.py`, `r2_oracle_cost.py`, `r2_weighted_agg.py`, `r2_rerun_mgh33.m`,
  `run_all_experiments.m`, `run_crit.m`, `weighted_run.m` --- resynchronised with that tree.

Added (`src/r3_postprocessing/`):

- `recompute_common_stationarity_r4.m` --- re-solve and certification ladder for the common proximal subproblem.
- `cert_sample.m` --- independent re-certification of a systematic sample of the first-pass values.
- `export_pareto_composite.m` --- composite objectives at the stored final points (Figure 1).
- `export_instances.m`, `r4_table4.py`, `r4_numbers.py`, `r4_audit.py` --- instance export, Table 4, and its audit.
- `run_r4.m`, `run_sample.m`, `run_exports.m`, `run_inst.m` --- drivers. Every path is resolved from the location of the script itself; the earlier wrappers contained absolute paths of the author's machine.
- `r4_table4.csv` --- the resulting table, as shipped.

## Added files

MATLAB:

- `EBNPG.m` — explicit Bregman nonmonotone proximal-gradient method (variants N0--N3, N5--N7)
- `EBNPG_W.m` — EBNPG with the adaptive BB-type diagonal variable metric
- `CNPG.m` — composite nonmonotone proximal-gradient baselines (max-type and average-type)
- `run_all_experiments.m` — master driver for the 42-problem principal experiment
- `run_problem.m` — single-problem driver (reproduces the stored starting-point
  assignment of all 42 problems, including the block maps of the four
  interrupted sessions AP4, BK1, PNR and DGO2; see the "Starting points"
  subsection of `README.md` for the protocol)
- `run_ebnpg_silent.m` — silent EBNPG wrapper
- `weighted_run.m` — adaptive EBNPG-W driver
- `noneuclid_run.m` — fixed non-Euclidean kernel study (paired starting points from the 8th stream block, the 7th for BK1 and PNR, reproducing the stored study)
- `recompute_common_stationarity.m` — common-stationarity re-evaluation
- `export_all_results.m` — aggregate stored runs into CSV files
- `export_archive.m` — export the reproducible instance data (applies the starting-point block map of `run_problem.m`)
- `qp_count_study.m`, `qp_count_full.m`, `qp_one.m` — proximal-solve counting
- `run_crit.m` — driver for the common-stationarity outputs
- `r2_rerun_mgh33.m` — documented MGH33 rerun
- `r2_qp_mgh33.m` — QP-solve counts for MGH33
- `r2_verify_k0.m` — consistency check of the MGH33 rerun
- `probe_euclid.m`, `probe_wratio.m` — diagnostics used during the revision

Python:

- `aggregate_all.py` — master aggregation of the manuscript tables
- `r2_weighted_agg.py` — Table 9 and Subsection 6.11 prose numbers
- `r2_oracle_cost.py` — per-iteration evaluation counts and weighted oracle cost
- `cmh_test.py`, `paired_test.py`, `mcnemar.py` — statistical tests
- `build_kernel.py`, `build_tab5.py`, `recompute_sr.py` — supporting aggregations
- `regen_summary.py` — rebuild `all_summary.csv`
- `r2_perf_profiles.py` — performance-profile figure
- `generate_pareto_plots.py` — Pareto-front figure

## Modified MPG files

- `ProxGrad.m`
- `armijo.m`
- `datas.m`
- `evalg.m`
- `evalgradg.m`
- `evalh.m`
- `explicitLS.m`
- `explicitLSone.m`
- `implicitLS.m`
- `inip.m` — problem set used in the manuscript, including the six new test problems
- `main.m` — demonstration driver for problem AP1 (ten algorithms over 100
  random starts). Two changes relative to the MPG original: the two
  ProxGradAcc test cases were removed, because ProxGradAcc is not part of
  this release and is not used by any experiment reported in the
  manuscript; and the interquartile ranges are computed with `prctile`
  instead of `iqr`, so that the release runs with the Optimization
  Toolbox alone
- `subproblemProxGrad.m`
