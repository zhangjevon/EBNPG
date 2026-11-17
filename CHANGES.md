# Changes relative to the MPG repository

This release is a modified version of <https://github.com/lfprudente/MPG>
(GNU GPL-3.0). The files below are grouped by their relation to that
repository. Files marked *modified* differ from the corresponding MPG file;
files marked *added* are new in this release.

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
