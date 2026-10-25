% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
% Single-problem batch runner for EBNPG experiments
% Usage: run_problem('AP1', 10)
% Results saved to results/<problem>_results.mat and results/<problem>_summary.csv
% Also saves final solutions x_final and objective values f_final for Pareto analysis
%
% Starting-point protocol (Subsection 6.2 of the manuscript): after the
% stream is seeded with 123 and the instance is generated, the twelve
% algorithms consume successive blocks of <num_runs> draws of
% x0 = l + (u-l).*rand(n,1), one block per algorithm.  In four of the
% original sessions the run was interrupted -- after the first algorithm
% for AP4, BK1 and PNR, after the third for DGO2 -- and the resumed session
% restarted the stream, so the first algorithm after the resume reused
% block 1 and every later algorithm moved one block earlier.  The block
% map below reproduces that assignment exactly, so the stored results of
% all 42 problems are regenerated bit for bit.

function run_problem(prob_name, num_runs)
    global problem
    
    if nargin < 2
        num_runs = 100;
    end
    
    rng(123);
    
    % Algorithm definitions
    algorithms = {};
    algorithms{end+1} = struct('name','MPG-Explicit','alg',1,'mode','','extra',[]);
    algorithms{end+1} = struct('name','MPG-Armijo',  'alg',2,'mode','','extra',[]);
    algorithms{end+1} = struct('name','MPG-Implicit','alg',3,'mode','','extra',[]);
    algorithms{end+1} = struct('name','CNPG-Max',    'alg',7,'mode','max','extra',[]);
    algorithms{end+1} = struct('name','CNPG-Avg',    'alg',7,'mode','avg','extra',[]);
    
    ebnpg_modes = {'N0','N1','N2','N3','N5','N6','N7'};
    for mi = 1:length(ebnpg_modes)
        algorithms{end+1} = struct('name',['EBNPG-' ebnpg_modes{mi}], ...
            'alg',6, 'mode',ebnpg_modes{mi}, ...
            'extra',struct('eta',0.5,'sigma',1e-4,'gamma',0.5,'beta',1.0,...
            'stop_tol',1e-4,'itermax',200,'M',5,'rho',0.9,'theta_avg',0.5));
    end
    
    % Create results directory
    if ~exist('results', 'dir')
        mkdir('results');
    end
    
    fprintf('===== Problem: %s (%d runs) =====\n', prob_name, num_runs);
    
    problem = prob_name;
    [n,m,l,u,x0_init] = inip;
    delta = (8*rand+2)/100 * norm(x0_init);
    [dimA,A,b] = datas(n,m,delta);
    
    fprintf('  n=%d, m=%d\n', n, m);
    
    % ---- starting-point blocks ---------------------------------------------
    % blk(a) is the index of the stream block consumed by algorithm slot a.
    % The blocks are drawn up front (the solvers consume no random numbers,
    % so this is equivalent to drawing each block when its algorithm runs)
    % and are therefore identical after a crash-resume.
    %
    % Natural protocol: slot a consumes block a.  The four problems whose
    % original sessions were interrupted and resumed without advancing the
    % stream deviate from it; the map below reproduces the stored assignment
    % of every problem exactly:
    %   AP4, BK1, PNR : interrupted after algorithm 1 -> slots 2..12 use blocks 1..11
    %   DGO2          : interrupted after algorithm 3 -> slots 4..12 use blocks 1..9
    blk = 1:length(algorithms);
    if ismember(prob_name, {'AP4','BK1','PNR'})
        blk = [1 1 2:(length(algorithms)-1)];
    elseif ismember(prob_name, {'DGO2'})
        blk = [1 2 3 1:9];
    end
    blocks = cell(length(algorithms), 1);
    for a = 1:length(algorithms)
        S = zeros(n, num_runs);
        for r = 1:num_runs
            S(:,r) = l + (u - l) .* rand(n, 1);
        end
        blocks{a} = S;
    end
    
    all_summary = {};
    prob_results = cell(length(algorithms), 1);

    % Check for existing partial results (crash recovery).  The starting-point
    % blocks are redrawn deterministically from the seed, so a resumed run
    % continues on exactly the same points as an uninterrupted one.
    partial_file = fullfile('results', [prob_name '_partial_results.mat']);
    start_alg = 1;
    if exist(partial_file, 'file')
        fprintf('  Found partial results, resuming from crash...\n');
        loaded = load(partial_file);
        prob_results = loaded.prob_results;
        % Rebuild summary and find resume point
        for aidx = 1:length(algorithms)
            if ~isempty(prob_results{aidx})
                r = prob_results{aidx};
                all_summary{end+1,1} = r.problem;
                all_summary{end,2} = r.algorithm;
                all_summary{end,3} = r.n;
                all_summary{end,4} = r.m;
                all_summary{end,5} = r.success_rate;
                all_summary{end,6} = r.med_iter;
                all_summary{end,7} = r.med_ngev;
                all_summary{end,8} = r.med_nhev;
                all_summary{end,9} = r.med_time;
                start_alg = aidx + 1;
            end
        end
        fprintf('  Resuming from algorithm %d/%d\n', start_alg, length(algorithms));
    end

    for aidx = start_alg:length(algorithms)
        alg = algorithms{aidx};
        fprintf('  [%2d/%2d] %-15s ... ', aidx, length(algorithms), alg.name);
        
        iter_vals = zeros(num_runs, 1);
        ngev_vals = zeros(num_runs, 1);
        nhev_vals = zeros(num_runs, 1);
        time_vals = zeros(num_runs, 1);
        info_vals = zeros(num_runs, 1);
        x_final = zeros(n, num_runs);
        f_final = zeros(m, num_runs);
        
        for run = 1:num_runs
            x0_run = blocks{blk(aidx)}(:, run);
            
            try
                if alg.alg == 1
                    [~, x_out, info, iter, ngev, nhev, time] = evalc('ProxGrad(n,m,l,u,x0_run,1,dimA,A,b)');
                elseif alg.alg == 2
                    [~, x_out, info, iter, ngev, nhev, time] = evalc('ProxGrad(n,m,l,u,x0_run,2,dimA,A,b)');
                elseif alg.alg == 3
                    [~, x_out, info, iter, ngev, nhev, time] = evalc('ProxGrad(n,m,l,u,x0_run,3,dimA,A,b)');
                elseif alg.alg == 6
                    e = alg.extra;
                    [x_out, info, iter, ngev, nhev, time] = ...
                        run_ebnpg_silent(n,m,l,u,x0_run,e.eta,e.sigma,e.gamma,e.beta,...
                            e.stop_tol,e.itermax,eye(n),A,b,dimA,alg.mode,e.M,e.rho,e.theta_avg);
                elseif alg.alg == 7
                    % CNPG: Composite Nonmonotone Proximal-Gradient
                    % (no evalc needed — CNPG is silent)
                    [x_out, info, iter, ngev, nhev, time] = ...
                        CNPG(n, m, l, u, x0_run, 1e-4, 0.5, 1e-4, 200, alg.mode, ...
                             A, b, dimA, 'M', 5, 'theta_avg', 0.5);
                end
                
                % Compute objective values at final solution
                f_out = zeros(m, 1);
                for j = 1:m
                    f_out(j) = evalg(n, x_out, j);
                end
            catch ME
                fprintf('ERROR: %s\n', ME.message);
                info = -99; iter = -1; ngev = -1; nhev = -1; time = -1;
                x_out = NaN(n,1);
                f_out = NaN(m,1);
            end
            
            iter_vals(run) = iter;
            ngev_vals(run) = ngev;
            nhev_vals(run) = nhev;
            time_vals(run) = time;
            info_vals(run) = info;
            x_final(:, run) = x_out;
            f_final(:, run) = f_out;
        end
        
        success_mask = (info_vals == 0);
        success_rate = sum(success_mask) / num_runs * 100;
        
        if sum(success_mask) > 0
            med_iter = median(iter_vals(success_mask));
            med_ngev = median(ngev_vals(success_mask));
            med_nhev = median(nhev_vals(success_mask));
            med_time = median(time_vals(success_mask));
        else
            med_iter = NaN; med_ngev = NaN; med_nhev = NaN; med_time = NaN;
        end
        
        fprintf('done (succ=%.0f%%, iter=%d, time=%.2fs)\n', ...
            success_rate, med_iter, med_time);
        
        result = struct();
        result.problem = prob_name;
        result.algorithm = alg.name;
        result.n = n; result.m = m;
        result.num_runs = num_runs;
        result.success_rate = success_rate;
        result.med_iter = med_iter;
        result.med_ngev = med_ngev;
        result.med_nhev = med_nhev;
        result.med_time = med_time;
        result.iter_vals = iter_vals;
        result.ngev_vals = ngev_vals;
        result.nhev_vals = nhev_vals;
        result.time_vals = time_vals;
        result.info_vals = info_vals;
        result.x_final = x_final;
        result.f_final = f_final;
        
        prob_results{aidx} = result;

        % Save intermediate results after each algorithm (crash recovery)
        partial_file = fullfile('results', [prob_name '_partial_results.mat']);
        save(partial_file, 'prob_results', 'prob_name');

        all_summary{end+1,1} = prob_name;
        all_summary{end,2} = alg.name;
        all_summary{end,3} = n;
        all_summary{end,4} = m;
        all_summary{end,5} = success_rate;
        all_summary{end,6} = med_iter;
        all_summary{end,7} = med_ngev;
        all_summary{end,8} = med_nhev;
        all_summary{end,9} = med_time;
    end
    
    % Save final results
    save(fullfile('results', [prob_name '_results.mat']), 'prob_results', 'prob_name');

    % Clean up partial results file
    partial_file = fullfile('results', [prob_name '_partial_results.mat']);
    if exist(partial_file, 'file')
        delete(partial_file);
    end

    summary_table = cell2table(all_summary, ...
        'VariableNames', {'Problem','Algorithm','n','m','SuccessRate','MedIter','MedNgev','MedNhev','MedTime'});
    writetable(summary_table, fullfile('results', [prob_name '_summary.csv']));
    
    fprintf('Results saved for %s\n\n', prob_name);
end
