# An Explicit Bregman Proximal-Gradient Method with Endogenous Nonmonotonicity for Composite Multiobjective Optimization

Code for the numerical experiments of the manuscript:

> *An explicit Bregman proximal-gradient method with endogenous nonmonotonicity for composite multiobjective optimization.*

This repository contains the **code**. The numerical outputs behind every table and figure of the
manuscript are archived in the data package `EBNPG_data_v3.1-jogo-r3.zip`, published as a release asset
of the tag `v3.1-jogo-r3` and archived on Zenodo (the DOI is given on the release page). The package is
self-contained: it carries the same code under `Code/` together with the stored runs under `Code_Data/`,
and it reproduces every table and figure without re-running any solver. No problem, run or data set
reported in the manuscript is withheld.

The code is a modified version of the MPG code of Bello--Cruz, Melo, Prudente and Serra
(<https://github.com/lfprudente/MPG>, GNU GPL-3.0); see `CHANGES.md` for the list of added and modified
files.

## Contents

All routines live in `src/`. The post-processing scripts resolve their inputs from their own location
by searching upwards for the folder that contains both `Code/` and `Code_Data/`, so inside the data
package they can be run from its root; the MATLAB drivers are run after adding `src/` (or `Code/`) to
the path.

### Main routine

- **`main.m`** — demonstration driver: selects the problem and the algorithm, runs ten algorithms over 100 random starting points and prints a summary table.

### Solvers

- **`EBNPG.m`** — explicit Bregman nonmonotone proximal-gradient method (variants N0--N3, N5--N7).
- **`EBNPG_W.m`** — EBNPG with the adaptive BB-type diagonal variable metric.
- **`CNPG.m`** — composite nonmonotone proximal-gradient baselines (max-type and average-type).
- **`ProxGrad.m`** — multiobjective proximal gradient (explicit, Armijo and implicit variants).

### Auxiliary routines

- **`inip.m`** — initial parameters of the test problems.
- **`evalg.m`**, **`evalgradg.m`**, **`evalh.m`**, **`datas.m`** — objective functions, their gradients, the constraints and their data.
- **`subproblemProxGrad.m`** — solver of the proximal subproblem.
- **`armijo.m`**, **`explicitLS.m`**, **`explicitLSone.m`**, **`implicitLS.m`** — line searches.

### Experiment drivers

- **`run_all_experiments.m`** — master driver of the principal experiment (42 problems).
- **`run_problem.m`** — single-problem driver; reproduces the stored starting-point assignment of all 42 problems (see its header comments for the protocol).
- **`weighted_run.m`** — adaptive EBNPG-W study.
- **`noneuclid_run.m`** — fixed non-Euclidean kernel study.
- **`qp_count_study.m`**, **`qp_count_full.m`**, **`qp_one.m`** — proximal (QP) solve counting.
- **`run_ebnpg_silent.m`**, **`run_crit.m`**, **`recompute_common_stationarity.m`** — supporting drivers.
- **`export_all_results.m`**, **`export_archive.m`** — export of stored runs and instance data.
- **`r2_rerun_mgh33.m`**, **`r2_qp_mgh33.m`**, **`r2_verify_k0.m`** — documented MGH33 rerun and its checks.
- **`probe_euclid.m`**, **`probe_wratio.m`** — diagnostics used during the revision.

### Post-processing (Python)

- **`aggregate_all.py`** — master aggregation of the manuscript tables.
- **`regen_summary.py`** — rebuild `all_summary.csv`.
- **`r2_weighted_agg.py`**, **`r2_oracle_cost.py`** — Table 9 and oracle-cost numbers.
- **`generate_perf_profiles.py`** — performance-profile figure (Figure 2 of the manuscript).
- **`generate_pareto_plots.py`** — Pareto-front figure (Figure 1), drawn in the plane of the composite
  objectives `F_i = f_i + g_i`. It reads `pareto_data_composite.csv`, produced by
  `r3_postprocessing/export_pareto_composite.m`; the older `results/pareto_data.csv` holds the smooth
  components only and is used as a fallback.
- **`r2_perf_profiles.py`** — earlier version of the performance-profile figure.
- **`cmh_test.py`**, **`paired_test.py`**, **`mcnemar.py`** — statistical tests.
- **`build_kernel.py`**, **`build_tab5.py`**, **`recompute_sr.py`** — supporting aggregations.

### Third-revision scripts (`r3_postprocessing/`)

These produce Table 4 and Figure 1 from the stored outputs.

- **`recompute_common_stationarity_r4.m`** — re-solves and certifies the common proximal subproblem at every point the first pass left unsolved.
- **`cert_sample.m`** — independent re-certification of a systematic sample of the first-pass values.
- **`export_pareto_composite.m`** — evaluates the composite objectives at the stored final points (no solver re-run).
- **`export_instances.m`** — exports the instance data (reference points, `B_i`, `delta`).
- **`r4_table4.py`**, **`r4_numbers.py`**, **`r4_audit.py`** — build Table 4 and audit it against the stored `cert` flags.
- **`run_r4.m`**, **`run_sample.m`**, **`run_exports.m`**, **`run_inst.m`** — drivers; every path is resolved from the location of the script itself, so they run from any checkout.
- **`r4_table4.csv`** — the resulting table, as shipped.

## Reproducing the numbers

All reported numbers come from the stored outputs in the data package. After unzipping it:

```bash
python Code_Data/aggregate_all.py              # rebuilds r2_tables.json from the stored runs
python Code/r3_postprocessing/r4_table4.py     # Table 4, from the certified stationarity table
python Code_Data/recompute_sr.py               # the six success-rate columns of Table 4
python Code_Data/cmh_test.py                   # the CMH test of Table 6
python Code/generate_pareto_plots.py           # Figure 1
python Code/generate_perf_profiles.py          # Figure 2
python Code/r3_postprocessing/check_table4.py  # every cell of Table 4 against the LaTeX source
```

`check_table4.py` and `verify_manuscript.py` also need the LaTeX source, which the data package does
not include; run them in the source tree of the manuscript, where they compare the printed table and
every prose claim with `r2_tables.json`. In the package they report that the sources are absent and
exit successfully.

`Code_Data/README.md` inside the package gives the full step order, the inputs of each step and the
cross-checks. `Code_Data/aggregate_all.py` reads the certified stationarity table `crit_out_r4/` by
default; set `EBPGM_CRIT=crit_out` only to reproduce the pre-revision state.

## Instructions

1. Add `src/` to the MATLAB path (or run from inside it).
2. For a quick demonstration, open `main.m`, select the problem and the algorithm, and run.
3. For the full experiments, run `run_all_experiments` and then post-process with the Python scripts listed above; `run_problem('AP1', 10)` is a small smoke test.

All experiments follow the fixed protocol of the manuscript: seed `123`, 42 test problems, 12 algorithms, 100 starting points per algorithm and problem. Generated outputs are written under `src/` and are ignored by git.

## Requirements

- MATLAB R2022b or later with Optimization Toolbox (`quadprog`, `linprog`)
- Python 3.8+ with `numpy`, `scipy`, `matplotlib`

## License

GNU GPL-3.0. See `LICENSE`. Added and modified files carry SPDX license headers; see `CHANGES.md`.
