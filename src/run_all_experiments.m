% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
% =============================================================================
% Reproduce All Numerical Experiments
% =============================================================================
% This script runs all 42 multiobjective test problems defined in inip.m.
%
% For each problem, 12 algorithms are tested with multiple random starts:
%   - MPG-Explicit, MPG-Armijo, MPG-Implicit (3 Euclidean baselines)
%   - CNPG-Max, CNPG-Avg (2 composite nonmonotone proximal-gradient)
%   - EBNPG-N0, N1, N2, N3, N5, N6, N7 (7 Bregman nonmonotone variants)
%
% Results are saved to results/ as:
%   - <problem>_results.mat  (full per-run data including Pareto solutions)
%   - <problem>_summary.csv  (median statistics)
%
% After all experiments complete, run:
%   1. export_all_results   (MATLAB) -> generates all_summary.csv, pareto_data.csv
%   2. r2_perf_profiles.py       (Python) -> Dolan-More performance profiles
%   3. generate_pareto_plots.py    (Python) -> Pareto front visualizations
%
% Usage:
%   >> run_all_experiments          % default: 100 runs per problem
%   >> run_all_experiments(10)      % 10 runs per problem
% =============================================================================

function run_all_experiments(num_runs)
    if nargin < 1
        num_runs = 100;
    end

    fprintf('==============================================================\n');
    fprintf('EBNPG Numerical Experiments - Full Reproduction\n');
    fprintf('Runs per problem: %d\n', num_runs);
    fprintf('==============================================================\n\n');

    % 42 test problems (the set of Table 1 of the manuscript)
    problem_list = { ...
        'AP1', 'AP3', 'AP4', 'BK1', 'DD1', 'DGO2', 'DTLZ2', 'FA1', 'Far1', ...
        'FDS', 'Hil1', 'IKK1', 'KW2', 'Lov4', 'Lov5', 'Lov6', 'LTDZ', ...
        'MGH16', 'MGH26', 'MGH33', 'MLF2', 'MMR1', 'MMR2', 'MMR3', 'MOP5', ...
        'MOP6', 'MOP7', 'NIC1', 'NIC2', 'NIC3', 'NIM1', 'NSP1', 'NSP2', ...
        'PNR', 'QV1', 'SK2', 'SP1', 'Toi4', 'Toi9', 'VU2', 'ZDT6', 'ZLT1' ...
    };

    n_problems = length(problem_list);
    fprintf('Total problems: %d\n', n_problems);
    fprintf('Algorithms per problem: 12\n');
    fprintf('Total runs: %d\n\n', n_problems * 12 * num_runs);

    % Create results directory
    if ~exist('results', 'dir')
        mkdir('results');
    end

    completed = 0;
    for i = 1:n_problems
        prob = problem_list{i};
        fprintf('\n[%d/%d] ', i, n_problems);

        % Skip if already completed
        matfile = fullfile('results', [prob '_results.mat']);
        if exist(matfile, 'file')
            fprintf('Skipping %s (already completed)\n', prob);
            completed = completed + 1;
            continue;
        end

        try
            run_problem(prob, num_runs);
            completed = completed + 1;
        catch ME
            fprintf('FAILED on %s: %s\n', prob, ME.message);
        end

        % Print progress
        fprintf('Progress: %d/%d completed\n', completed, n_problems);
    end

    fprintf('\n==============================================================\n');
    fprintf('All experiments complete! %d/%d problems finished.\n', completed, n_problems);
    fprintf('==============================================================\n');
    fprintf('\nNext steps:\n');
    fprintf('  1. Run in MATLAB: export_all_results\n');
    fprintf('  2. Run in Python: python generate_perf_profiles.py\n');
    fprintf('  3. Run in Python: python generate_pareto_plots.py\n');
    fprintf('  4. Run in Python: python update_manuscript_full.py\n');
end
