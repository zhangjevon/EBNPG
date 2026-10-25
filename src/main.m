% Modified from the MPG repository (https://github.com/lfprudente/MPG); see CHANGES.md.
% SPDX-License-Identifier: GPL-3.0-or-later
clear; clc;
global problem

rng(123);  % fixed random seed

% Select the problem
problem = 'AP1';

% Initial parameters
[n,m,l,u,x0] = inip;
delta = (8*rand+2)/100 * norm(x0);
[dimA,A,b] = datas(n,m,delta);

% ========== Mode selection ==========
% Whether to run the multi-start test
do_multirun = true;   % true: multi-start test, false: single run
num_runs = 100;       % number of random starting points (only if do_multirun=true)

% Algorithm choice for a single run (only if do_multirun=false)
% AlgOpt = 1: Explicit Multiobjective Proximal Gradient
% AlgOpt = 2: Multiobjective Proximal Gradient with Armijo line search
% AlgOpt = 3: Implicit Multiobjective Proximal Gradient
% AlgOpt = 6: EBNPG (several nonmonotone modes, only for do_multirun=true)
AlgOpt = 1;  % example; can be changed to 1-3

% ====================================

if ~do_multirun
    % ---------- Single-run mode ----------
    if ( AlgOpt == 1 )
        [x, info, iter, ngev, nhev, time] = ProxGrad(n,m,l,u,x0,1,dimA,A,b);
    elseif ( AlgOpt == 2 )
        [x, info, iter, ngev, nhev, time] = ProxGrad(n,m,l,u,x0,2,dimA,A,b);
    elseif ( AlgOpt == 3 )
        [x, info, iter, ngev, nhev, time] = ProxGrad(n,m,l,u,x0,3,dimA,A,b);
    else
        error('Invalid AlgOpt for single run. Choose 1-3.');
    end
    % The results are already printed inside the solver; nothing else to do
    
else
    % ---------- Multi-start test mode ----------
    % Define the list of algorithms to test
    test_cases = struct('name', {}, 'alg', {}, 'mode', {}, 'extra', {});
    
    % Add the ProxGrad variants (AlgOpt = 1-3)
    test_cases(end+1).name = 'ProxGrad-Explicit'; 
    test_cases(end).alg = 1; 
    test_cases(end).mode = []; 
    test_cases(end).extra = [];
    
    test_cases(end+1).name = 'ProxGrad-Armijo';   
    test_cases(end).alg = 2; 
    test_cases(end).mode = []; 
    test_cases(end).extra = [];
    
    test_cases(end+1).name = 'ProxGrad-Implicit'; 
    test_cases(end).alg = 3; 
    test_cases(end).mode = []; 
    test_cases(end).extra = [];
    
    % Add the EBNPG modes (N0~N7)
    modes_ebnpg = {'N0','N1','N2','N3','N5','N6','N7'};
    for mi = 1:length(modes_ebnpg)
        test_cases(end+1).name = ['EBNPG-' modes_ebnpg{mi}];
        test_cases(end).alg = 6;
        test_cases(end).mode = modes_ebnpg{mi};
        test_cases(end).extra = struct(...
            'eta', 0.5, ...
            'sigma', 1e-4, ...
            'gamma', 0.5, ...
            'beta', 1.0, ...
            'stop_tol', 1e-4, ...
            'itermax', 200, ...
            'Q', eye(n), ...
            'M', 5, ...
            'rho', 0.9, ...
            'theta_avg', 0.5 ...
        );
    end
    
    % Preallocate the result storage
    all_results = cell(length(test_cases),1);
    
    for tc = 1:length(test_cases)
        case_info = test_cases(tc);
        fprintf('\n========== Testing %s over %d runs ==========\n', case_info.name, num_runs);
        
        iter_vals = zeros(num_runs,1);
        ngev_vals = zeros(num_runs,1);
        nhev_vals = zeros(num_runs,1);
        time_vals = zeros(num_runs,1);
        info_vals = zeros(num_runs,1);
        
        for run = 1:num_runs
            % Draw the starting point uniformly from [l,u]
            x0_run = l + (u - l) .* rand(n,1);
            
            % Call the corresponding solver according to alg (all solvers return the full set of outputs)
            if case_info.alg == 1
                [~, info, iter, ngev, nhev, time] = ProxGrad(n,m,l,u,x0_run,1,dimA,A,b);
            elseif case_info.alg == 2
                [~, info, iter, ngev, nhev, time] = ProxGrad(n,m,l,u,x0_run,2,dimA,A,b);
            elseif case_info.alg == 3
                [~, info, iter, ngev, nhev, time] = ProxGrad(n,m,l,u,x0_run,3,dimA,A,b);
            elseif case_info.alg == 6
                e = case_info.extra;
                [~, info, iter, ngev, nhev, time] = EBNPG(n,m,l,u,x0_run, ...
                    e.eta, e.sigma, e.gamma, e.beta, e.stop_tol, e.itermax, ...
                    e.Q, A, b, dimA, case_info.mode, ...
                    'M', e.M, 'rho', e.rho, 'theta_avg', e.theta_avg);
            else
                error('Unknown algorithm');
            end
            
            iter_vals(run) = iter;
            ngev_vals(run) = ngev;
            nhev_vals(run) = nhev;
            time_vals(run) = time;
            info_vals(run) = info;
        end
        
        % Compute the statistics
        results = struct();
        results.name = case_info.name;
        results.iter_median = median(iter_vals);
        results.ngev_median = median(ngev_vals);
        results.nhev_median = median(nhev_vals);
        results.time_median = median(time_vals);
        results.success_rate = sum(info_vals == 0) / num_runs * 100;
        results.iter_iqr = prctile(iter_vals,75) - prctile(iter_vals,25);
        results.time_iqr = prctile(time_vals,75) - prctile(time_vals,25);
        all_results{tc} = results;
    end
    
    % Print the summary table
    fprintf('\n========== Summary over %d random starts ==========\n', num_runs);
    fprintf('%-22s %-10s %-10s %-10s %-10s %-10s\n', ...
        'Algorithm', 'iter(med)', 'ngev(med)', 'nhev(med)', 'time(med)', 'success%');
    for tc = 1:length(all_results)
        r = all_results{tc};
        fprintf('%-22s %-10d %-10d %-10d %-10.2f %-10.1f\n', ...
            r.name, r.iter_median, r.ngev_median, r.nhev_median, r.time_median, r.success_rate);
    end
end