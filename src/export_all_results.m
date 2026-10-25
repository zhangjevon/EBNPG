% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
% run this script from the src folder
results_dir = 'results';
diary('export_log.txt');

% Get all .mat files (both _results.mat and _partial_results.mat)
files = dir(fullfile(results_dir, '*.mat'));
fprintf('Found %d mat files\n', length(files));

% Sort files alphabetically
[~, sortidx] = sort({files.name});
files = files(sortidx);

% ============================================================
% 1. Export summary data (medians) to single CSV
% ============================================================
summary_data = {};
count = 0;
for fi = 1:length(files)
    prob_name = files(fi).name;
    prob_name = strrep(prob_name, '_results.mat', '');
    prob_name = strrep(prob_name, '_partial_results.mat', '');
    
    data = load(fullfile(results_dir, files(fi).name));
    
    if ~isfield(data, 'prob_results')
        fprintf('WARNING: %s has no prob_results field, skipping\n', files(fi).name);
        continue;
    end
    
    pr = data.prob_results;
    
    for ai = 1:length(pr)
        r = pr{ai};
        if ~isstruct(r), continue; end  % Skip empty entries in partial results
        count = count + 1;
        summary_data{count,1} = prob_name;
        summary_data{count,2} = r.algorithm;
        summary_data{count,3} = r.n;
        summary_data{count,4} = r.m;
        summary_data{count,5} = r.num_runs;
        summary_data{count,6} = r.success_rate;
        summary_data{count,7} = r.med_iter;
        summary_data{count,8} = r.med_ngev;
        summary_data{count,9} = r.med_nhev;
        summary_data{count,10} = r.med_time;
    end
    fprintf('  Processed %s (%d algorithms)\n', prob_name, length(pr));
end

summary_table = cell2table(summary_data, ...
    'VariableNames', {'Problem','Algorithm','n','m','NumRuns','SuccessRate','MedIter','MedNgev','MedNhev','MedTime'});
writetable(summary_table, fullfile(results_dir, 'all_summary.csv'));
fprintf('Summary exported: %d rows\n', height(summary_table));

% ============================================================
% 2. Export per-run raw data for performance profiles
% ============================================================
raw_data = {};
count = 0;
for fi = 1:length(files)
    prob_name = files(fi).name;
    prob_name = strrep(prob_name, '_results.mat', '');
    prob_name = strrep(prob_name, '_partial_results.mat', '');
    
    data = load(fullfile(results_dir, files(fi).name));
    
    if ~isfield(data, 'prob_results')
        continue;
    end
    
    pr = data.prob_results;
    
    for ai = 1:length(pr)
        r = pr{ai};
        if ~isstruct(r), continue; end  % Skip empty entries in partial results
        for run_idx = 1:r.num_runs
            count = count + 1;
            raw_data{count,1} = prob_name;
            raw_data{count,2} = r.algorithm;
            raw_data{count,3} = run_idx;
            raw_data{count,4} = r.n;
            raw_data{count,5} = r.m;
            raw_data{count,6} = r.info_vals(run_idx);
            raw_data{count,7} = r.iter_vals(run_idx);
            raw_data{count,8} = r.ngev_vals(run_idx);
            raw_data{count,9} = r.nhev_vals(run_idx);
            raw_data{count,10} = r.time_vals(run_idx);
        end
    end
    fprintf('  Raw data for %s done\n', prob_name);
end

raw_table = cell2table(raw_data, ...
    'VariableNames', {'Problem','Algorithm','Run','n','m','Info','Iter','Ngev','Nhev','Time'});
writetable(raw_table, fullfile(results_dir, 'all_raw.csv'));
fprintf('Raw data exported: %d rows\n', height(raw_table));

% ============================================================
% 3. Export Pareto front data (f_final) for 2-objective problems
% ============================================================
pareto_data = {};
count = 0;
for fi = 1:length(files)
    prob_name = files(fi).name;
    prob_name = strrep(prob_name, '_results.mat', '');
    prob_name = strrep(prob_name, '_partial_results.mat', '');
    
    data = load(fullfile(results_dir, files(fi).name));
    
    if ~isfield(data, 'prob_results')
        continue;
    end
    
    pr = data.prob_results;
    
    for ai = 1:length(pr)
        r = pr{ai};
        if ~isstruct(r), continue; end  % Skip empty entries in partial results
        if r.m == 2 && isfield(r, 'f_final')
            for run_idx = 1:r.num_runs
                if r.info_vals(run_idx) == 0  % only successful runs
                    count = count + 1;
                    pareto_data{count,1} = prob_name;
                    pareto_data{count,2} = r.algorithm;
                    pareto_data{count,3} = run_idx;
                    pareto_data{count,4} = r.f_final(1, run_idx);
                    pareto_data{count,5} = r.f_final(2, run_idx);
                end
            end
        end
    end
end

if ~isempty(pareto_data)
    pareto_table = cell2table(pareto_data, ...
        'VariableNames', {'Problem','Algorithm','Run','f1','f2'});
    writetable(pareto_table, fullfile(results_dir, 'pareto_data.csv'));
    fprintf('Pareto data exported: %d points\n', height(pareto_table));
else
    fprintf('No 2-objective problems found for Pareto export\n');
end

fprintf('\nAll exports complete.\n');
diary('off');
