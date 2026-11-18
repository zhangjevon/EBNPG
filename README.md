# EBNPG numerical implementation

Companion code for *An explicit Bregman proximal-gradient method with endogenous
nonmonotonicity for composite multiobjective optimization*, by Jian-Wen Peng,
Jie-Wen Zhang, Orizon P. Ferreira and Jen-Chih Yao.

## Final numerical revision: v3.4-jogo-r3

This revision completes the common-displacement comparison at the stored final
iterates. The accompanying versioned data archive contains all 42 problems and
50,400 principal runs, the documented twelve-algorithm MGH33 rerun, adaptive and
fixed-kernel outputs, instance data and figures. No main optimization run was
repeated for this final-point verification.

At the 134 points requiring additional checks, 31 satisfy both common criteria
and 103 satisfy neither. The exact per-point witnesses are stored in
`Code_Data/crit_out_r4/adjudication_witnesses.json`. Candidate solvers only
propose feasible points. Exact rational arithmetic evaluates the exported
binary64 coefficient model, and integer bounds enclose square roots. Original
transcendental derivatives before numerical export are not claimed to be exact.
Every reported tolerance is applied to its own interval; an undecided result
stops aggregation instead of counting as a failed optimization run.

The data archive `EBNPG_data_v3.4-jogo-r3.zip` is prepared separately from this
code-only repository. Its metadata records this code snapshot, the frozen
baseline archive and file checksums. Publication status and an optional version
DOI are recorded by the release process; no new DOI is assumed here.

## Layout

`src/` contains the MATLAB algorithms, test functions and Python aggregation
routines. `src/r3_postprocessing/` contains final-point export, certificate,
independent replay and Table 4 routines. In the data archive these are installed
under `Code/`, `Code/r3_postprocessing/` and `Code_Data/` as documented there.
The original algorithms and test inputs remain as specified in the manuscript.

## Reproduction from the data archive

From a clean unpack of the archive:

```text
python Code/r3_postprocessing/replay_adjudication.py Code_Data/crit_out_r4/adjudicated.csv Code_Data/crit_out_r4/adjudication_witnesses.json replay_report.json
python Code_Data/recompute_sr.py
python Code_Data/aggregate_all.py
python Code/r3_postprocessing/r4_table4.py Code_Data/crit_out_r4
```

Independent witness replay uses only Python's standard library. Aggregation and
candidate generation require NumPy and SciPy; QP candidates use OSQP. Exact
versions tested in this revision are recorded in `tested_environment.json`
inside the archive. Its shipped final witnesses allow the tables to be checked
without rerunning either the optimization algorithms or the candidate solvers.

To regenerate certificate candidates, use the 11 archived formal MAT exports
with `certify_adjudicated.py`; see its command-line help and the archive README.
Regenerated valid bounds can differ between solver versions while deciding the
same criteria. A new MATLAB export can have a different file hash because MAT
headers include a timestamp; coefficient checks are distinct from file hashes.

`verify_manuscript.py --tex <source.tex> --data <Code_Data>` checks the tables and
numerical statements against the final outputs. Missing source/data are errors.
`check_table4.py <source.tex> <common_sr.json>` checks all 72 printed cells.
The manuscript sources are distributed with the journal revision, separately
from the numerical archive.

## Attribution and licence

The implementation modifies the MPG code of Bello-Cruz, Melo, Prudente and Serra
(https://github.com/lfprudente/MPG). See `CHANGES.md` for this revision and the
historical changes. Code is distributed under GPL-3.0-or-later; see `LICENSE`.
Historical sections of CHANGES describe superseded methods and are not the
current reproduction instructions.
