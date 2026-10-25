% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function recompute_common_stationarity(prob_list, out_dir)
% Recompute a COMMON, method-independent stationarity measure at every
% stored final iterate.  No solver is re-run: this is pure post-processing
% of the stored x_final.
%
% For each problem / algorithm / run we solve the proximal subproblem ONCE
% with the SAME fixed proximal parameter beta = 1 and the SAME Euclidean
% kernel Q = I, and record
%     crit  = -Theta_1(x)      (common criticality measure, >= 0)
%     dnorm = ||p_1(x) - x||   (common proximal displacement)
%
% Usage:  recompute_common_stationarity({'AP1','BK1'}, 'crit_out')

    if nargin < 2
        out_dir = 'crit_out';
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    global problem

    for pi = 1:length(prob_list)
        prob_name = prob_list{pi};
        t0 = tic;

        % ---- reproduce EXACTLY the random objects of run_problem.m --------
        rng(123);
        problem = prob_name;
        [n, m, l, u, x0_init] = inip;
        delta = (8*rand + 2)/100 * norm(x0_init);
        [dimA, A, b] = datas(n, m, delta);

        % closed-form B_i for the support function: g_i(x) = delta*||B_i^{-T}x||_1
        Bcell = cell(m,1);
        for i = 1:m
            Bcell{i} = A{i}(1:n, :);
        end

        % ---- load stored results -----------------------------------------
        matfile = fullfile('results', [prob_name '_results.mat']);
        if ~exist(matfile, 'file')
            fprintf('MISSING %s\n', matfile);
            continue
        end
        loaded = load(matfile);
        pr = loaded.prob_results;
        nalg = length(pr);

        out = struct();
        out.problem = prob_name;
        out.n = n; out.m = m; out.delta = delta;
        out.algorithm = cell(nalg,1);
        out.crit   = nan(nalg, 100);   % -Theta_1(x_final)
        out.dnorm  = nan(nalg, 100);   % ||p_1(x_final) - x_final||
        out.info   = nan(nalg, 100);   % original solver flag (0 = success)
        out.iter   = nan(nalg, 100);
        out.ngev   = nan(nalg, 100);
        out.nhev   = nan(nalg, 100);
        out.time   = nan(nalg, 100);

        Q = eye(n);
        beta = 1;

        for a = 1:nalg
            r = pr{a};
            if isempty(r)
                continue
            end
            out.algorithm{a} = r.algorithm;
            xf = r.x_final;            % n x 100
            nruns = size(xf, 2);
            for run = 1:nruns
                x = xf(:, run);
                out.info(a,run) = r.info_vals(run);
                out.iter(a,run) = r.iter_vals(run);
                out.ngev(a,run) = r.ngev_vals(run);
                out.nhev(a,run) = r.nhev_vals(run);
                out.time(a,run) = r.time_vals(run);
                if any(~isfinite(x))
                    continue
                end
                JG = zeros(m, n);
                H  = zeros(m, 1);
                ok = true;
                for i = 1:m
                    JG(i,:) = evalgradg(n, x, i);
                    if any(~isfinite(JG(i,:)))
                        ok = false; break
                    end
                    H(i) = delta * norm(Bcell{i}' \ x, 1);
                    if ~isfinite(H(i))
                        ok = false; break
                    end
                end
                if ~ok
                    continue
                end
                [theta, ~, p, flag] = subproblemProxGrad(n, m, l, u, x, beta, JG, H, dimA, A, b, Q);
                if flag == 0
                    out.crit(a,run)  = -theta;
                    out.dnorm(a,run) = norm(p - x);
                end
            end
        end

        save(fullfile(out_dir, [prob_name '_crit.mat']), 'out');
        fprintf('%-6s done in %6.1f s\n', prob_name, toc(t0));
    end
end
