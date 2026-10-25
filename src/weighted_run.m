% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function weighted_run(probs, epsx, nruns, outdir)
% weighted_run -- reconstruct the adaptive EBNPG-W experiment of Subsection 6.11.
%
% Usage (from the Code folder):
%   weighted_run({'NIC1'}, 1e-8, 100, 'results_w_rebuilt')
%
% For every problem: rng(123); [n,m,l,u,x0_init]=inip; delta=(8*rand+2)/100*||x0_init||;
% [dimA,A,b]=datas(n,m,delta); then the seven EBNPG-W variants N0,N1,N2,N3,N5,N6,N7
% are run in that order, each on its own block of <nruns> draws of
%   x0 = l + (u-l).*rand(n,1)
% from the same stream (the protocol of run_problem.m).
%
% Results are written per problem as <prob>_W_results.mat (cell 7x1, same field
% names as run_problem.m) and <prob>_W_summary.csv, so the stored and the
% rebuilt sets can be compared field by field.

if nargin < 2 || isempty(epsx),  epsx = 1e-8;   end
if nargin < 3 || isempty(nruns), nruns = 100;   end
if nargin < 4 || isempty(outdir),outdir = 'results_w_rebuilt'; end
if ~exist(outdir,'dir'), mkdir(outdir); end

MODES = {'N0','N1','N2','N3','N5','N6','N7'};

for pi = 1:numel(probs)
    prob_name = probs{pi};
    outfile = fullfile(outdir,[prob_name '_W_results.mat']);
    if exist(outfile,'file')
        fprintf('%-6s already done, skipping\n', prob_name);
        continue
    end

    clear global problem
    global problem

    rng(123);
    problem = prob_name;
    [n,m,l,u,x0_init] = inip;                                   %#ok<ASGLU>
    delta = (8*rand+2)/100 * norm(x0_init);
    [dimA,A,b] = datas(n,m,delta);

    % run-level crash recovery
    partfile = fullfile(outdir,[prob_name '_W_partial.mat']);
    prob_results = cell(numel(MODES),1);
    start_m = 1;
    if exist(partfile,'file')
        ld = load(partfile);
        prob_results = ld.prob_results;
        start_m = ld.next_m;
        nskip = (start_m-1)*nruns;
        for s = 1:nskip
            x0r = l + (u-l).*rand(n,1);                          %#ok<NASGU>
        end
        fprintf('  resuming at variant %d, stream advanced by %d draws\n', start_m, nskip);
    end

    fprintf('===== %s (n=%d, m=%d, delta=%.6g) =====\n', prob_name, n, m, delta);

    for mi = start_m:numel(MODES)
        mode = MODES{mi};
        iter_vals = zeros(nruns,1);
        ngev_vals = zeros(nruns,1);
        nhev_vals = zeros(nruns,1);
        time_vals = zeros(nruns,1);
        info_vals = zeros(nruns,1);
        x_final   = zeros(n,nruns);
        f_final   = zeros(m,nruns);
        t0 = tic;

        for run = 1:nruns
            x0_run = l + (u - l) .* rand(n,1);
            try
                [x_out, info, iter, ngev, nhev, time] = EBNPG_W( ...
                    n,m,l,u,x0_run,0.5,1e-4,0.5,1.0,1e-4,200,A,b,dimA,mode,epsx, ...
                    'M',5,'rho',0.9,'theta_avg',0.5);
                f_out = zeros(m,1);
                for j = 1:m
                    f_out(j) = evalg(n, x_out, j);
                end
            catch ME
                fprintf('    ERROR: %s\n', ME.message);
                info = -99; iter = -1; ngev = -1; nhev = -1; time = -1;
                x_out = NaN(n,1);
                f_out = NaN(m,1);
            end
            iter_vals(run) = iter;
            ngev_vals(run) = ngev;
            nhev_vals(run) = nhev;
            time_vals(run) = time;
            info_vals(run) = info;
            x_final(:,run)  = x_out;
            f_final(:,run)  = f_out;
        end

        success_mask = (info_vals == 0);
        success_rate = sum(success_mask)/nruns*100;
        if sum(success_mask) > 0
            med_iter = median(iter_vals(success_mask));
            med_ngev = median(ngev_vals(success_mask));
            med_nhev = median(nhev_vals(success_mask));
            med_time = median(time_vals(success_mask));
        else
            med_iter = NaN; med_ngev = NaN; med_nhev = NaN; med_time = NaN;
        end

        r = struct();
        r.problem = prob_name;
        r.algorithm = ['EBNPG-W-' mode];
        r.n = n; r.m = m;
        r.num_runs = nruns;
        r.success_rate = success_rate;
        r.med_iter = med_iter;
        r.med_ngev = med_ngev;
        r.med_nhev = med_nhev;
        r.med_time = med_time;
        r.iter_vals = iter_vals;
        r.ngev_vals = ngev_vals;
        r.nhev_vals = nhev_vals;
        r.time_vals = time_vals;
        r.info_vals = info_vals;
        r.x_final = x_final;
        r.f_final = f_final;
        prob_results{mi} = r;

        next_m = mi + 1;                                        %#ok<NASGU>
        save(partfile,'prob_results','next_m','prob_name');

        fprintf('  [%d/7] %-12s SR=%5.1f%%  it=%s ng=%s nh=%s  (%.1fs wall)\n', ...
            mi, r.algorithm, success_rate, ...
            fmt(med_iter), fmt(med_ngev), fmt(med_nhev), toc(t0));
    end

    save(outfile,'prob_results','prob_name');
    if exist(partfile,'file'), delete(partfile); end

    S = cell(numel(MODES),9);
    for mi = 1:numel(MODES)
        r = prob_results{mi};
        S{mi,1} = r.problem;  S{mi,2} = r.algorithm;
        S{mi,3} = r.n;        S{mi,4} = r.m;
        S{mi,5} = r.success_rate; S{mi,6} = r.med_iter; S{mi,7} = r.med_ngev;
        S{mi,8} = r.med_nhev;    S{mi,9} = r.med_time;
    end
    T = cell2table(S,'VariableNames', ...
        {'Problem','Algorithm','n','m','SuccessRate','MedIter','MedNgev','MedNhev','MedTime'});
    writetable(T, fullfile(outdir,[prob_name '_W_summary.csv']));
    fprintf('  saved %s\n\n', outfile);
end
fprintf('ALL DONE\n');
end

function s = fmt(v)
    if isnan(v), s = 'NaN'; else, s = sprintf('%.1f', v); end
end
