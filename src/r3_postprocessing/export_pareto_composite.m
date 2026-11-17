% Added for the EBNPG manuscript (third revision).
% SPDX-License-Identifier: GPL-3.0-or-later
function export_pareto_composite(prob_list, out_csv)
% EXPORT_PARETO_COMPOSITE  Composite objective values for the Pareto figure.
%
% The solver stores, for every run, the final point x and the *smooth*
% values f_i(x) (run_problem.m, line "f_out(j) = evalg(n,x_out,j)").  The
% Pareto front of problem (1) is the image of F_i = f_i + g_i, not of f_i,
% so the figure has to be drawn from
%
%   F_i(x) = f_i(x) + delta * || B_i^{-T} x ||_1 ,      B_i = A{i}(1:n,:) ,
%
% with A{i}, b{i} rebuilt exactly as in inip.m / datas.m (rng(123) replay,
% the same construction the certification drivers use) and g_i evaluated in
% closed form.  The closed form is cross-checked against evalh.m, the LP the
% solver itself uses, on the first finite point of every problem.
%
% No solver is re-run: the points are the stored ones, restricted to the runs
% the algorithm declared successful (info_vals == 0, the mask used by
% run_problem.m), which is what the figure is described as showing.
%
% Usage:  export_pareto_composite({'BK1','DD1','DGO2','AP3','Hil1','Far1'})
% Output: Code_Data/pareto_data_composite.csv  (Problem,Algorithm,Run,F1,F2)

if nargin < 1 || isempty(prob_list)
    prob_list = {'BK1','DD1','DGO2','AP3','Hil1','Far1'};
end
if nargin < 2 || isempty(out_csv)
    out_csv = 'pareto_data_composite.csv';
end

root     = pkg_root(mfilename('fullpath'));   % folder holding Code/ and Code_Data/
code_dir = fullfile(root,'Code');
data_dir = fullfile(root,'Code_Data');
old_pwd  = pwd; cd(data_dir); cleanup = onCleanup(@() cd(old_pwd)); %#ok<NASGU>
addpath(code_dir);
global problem

fid = fopen(out_csv,'w');
fprintf(fid,'Problem,Algorithm,Run,F1,F2\n');
ntot = 0;

for pi = 1:numel(prob_list)
    p = prob_list{pi};

    % ---- rebuild the instance exactly as run_problem.m did ----------------
    rng(123); problem = p;
    [n,m,l,u,x0_init] = inip;
    delta = (8*rand + 2)/100 * norm(x0_init);
    [dimA,A,b] = datas(n,m,delta);
    Bc = cell(m,1);
    for i = 1:m, Bc{i} = A{i}(1:n,:); end

    loaded = load(fullfile('results',[p '_results.mat']));
    pr = loaded.prob_results; nalg = numel(pr);
    nrun = 0; nbox = 0; hchk = NaN;

    for a = 1:nalg
        r = pr{a}; if isempty(r), continue; end
        X = r.x_final; IV = r.info_vals;
        nsu = 0;
        for run = 1:size(X,2)
            x = X(:,run);
            if any(~isfinite(x)), continue; end
            % the figure shows successful runs only (info == 0, as in
            % run_problem.m: success_mask = (info_vals == 0))
            if numel(IV) >= run && IV(run) ~= 0, continue; end
            f = zeros(m,1); g = zeros(m,1);
            for i = 1:m
                f(i) = evalg(n,x,i);
                g(i) = delta * norm(Bc{i}'\x, 1);
            end
            F = f + g;
            if any(~isfinite(F)) || m < 2, continue; end
            % box feasibility of the plotted point
            if any(x < l - 1e-12) || any(x > u + 1e-12), nbox = nbox + 1; end
            % cross-check the closed form of g_1 against the LP of evalh.m
            if isnan(hchk)
                [hlp,infoh] = evalh(n,x,1,A,b);
                if infoh == 0, hchk = abs(hlp - g(1)); end
            end
            fprintf(fid,'%s,%s,%d,%.12g,%.12g\n',p,r.algorithm,run,F(1),F(2));
            nrun = nrun + 1;
        end
    end
    ntot = ntot + nrun;
    fprintf('%-6s m=%d  points=%4d  box-violations=%d  |evalh-closed|=%8.2e\n', ...
            p,m,nrun,nbox,hchk);
end

fclose(fid);
fprintf('\nwritten %s : %d points over %d problems\n', out_csv, ntot, numel(prob_list));
end

% ------------------------------------------------------------------------
function root = pkg_root(this_file)
% The folder that contains both Code/ and Code_Data/, found by moving upwards.
% This makes the driver independent of where it sits: in the development tree
% it lives in patch/, in the release package in Code/r3_postprocessing/, and a
% fixed number of fileparts() calls cannot be right in both layouts.
root = fileparts(this_file);
while ~(exist(fullfile(root,'Code'),'dir') && exist(fullfile(root,'Code_Data'),'dir'))
    up = fileparts(root);
    if isempty(up) || strcmp(up,root)
        error('pkg_root:notfound', ...
              'no folder containing Code/ and Code_Data/ above %s', this_file);
    end
    root = up;
end
end
