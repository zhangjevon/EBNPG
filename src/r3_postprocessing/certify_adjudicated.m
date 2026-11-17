% Added for the EBNPG manuscript (third revision).
% SPDX-License-Identifier: GPL-3.0-or-later
function certify_adjudicated()
% CERTIFY_ADJUDICATED  Step 1 of 2: export the disputed subproblems.
%
% For every triple that needs additional verification -- the 128 listed in
% Code_Data/crit_out_r4/adjudicated_uncertified.csv plus the six at which the
% numeric solve of the subproblem in retry_log.csv returned no verified
% solution (method = -1) -- this script reproduces the instance exactly as
% run_problem.m does, rebuilds the common proximal subproblem at the stored
% final iterate
%
%   Theta(x,d) = max_i { grad f_i(x)'d + h_i(x+d) - h_i(x) } + 1/2 ||d||^2,
%   h_i(y)     = delta * || B_i^{-T} y ||_1 ,   B_i = A{i}(1:n,:),
%
% and writes its data (JG, H, the box, and the LP blocks A{i}, b{i}) to
% Code_Data/crit_out_r4/qp_export/<problem>.mat, for certify_adjudicated.py.
%
% The six are exported because the subproblem is 1-strongly convex on a closed
% box and therefore has a unique minimiser: the earlier record had no verified
% solution for them, not a subproblem without one, and they are decided by the
% same primal-dual bounds.
%
% Step 2 is
%   python Code/r3_postprocessing/certify_adjudicated.py \
%          Code_Data/crit_out_r4/qp_export \
%          Code_Data/crit_out_r4/adjudicated_uncertified.csv \
%          Code_Data/crit_out_r4/adjudicated.csv
% which brackets the minimiser of each subproblem with a primal-feasible upper
% bound and a dual-feasible lower bound and writes the per-point record.
%
% Usage (from anywhere):  certify_adjudicated
%
% The ladder in recompute_common_stationarity_r4.m is *not* used here: the
% whole point of this step is an independent solver and a certificate.

root = pkg_root(mfilename('fullpath'));
code_dir = fullfile(root,'Code');
data_dir = fullfile(root,'Code_Data');
crit_dir = fullfile(data_dir,'crit_out_r4');
res_dir  = fullfile(data_dir,'results');
out_dir  = fullfile(crit_dir,'qp_export');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

old_pwd = pwd; cd(data_dir); cleanup = onCleanup(@() cd(old_pwd));
addpath(code_dir);
addpath(fullfile(code_dir,'r3_postprocessing'));
global problem

fid = fopen(fullfile(crit_dir,'adjudicated_uncertified.csv'),'r');
if fid < 0
    error('adjudicated_uncertified.csv not found in %s', crit_dir);
end
fgetl(fid);
C = textscan(fid,'%s %s %d %*[^\n]','Delimiter',',');
fclose(fid);
probs = C{1}; algs = C{2}; runs = C{3};

% The six points at which the numeric solve of the subproblem returned no
% verified solution (method = -1 in retry_log.csv) belong in the same record.
fid = fopen(fullfile(crit_dir,'retry_log.csv'),'r');
if fid >= 0
    fgetl(fid);
    D = textscan(fid,'%s %s %d %d %d %*[^\n]','Delimiter',',');
    fclose(fid);
    take = (D{5} == -1);
    if any(take)
        probs = [probs; D{1}(take)];
        algs  = [algs;  D{2}(take)];
        runs  = [runs;  D{3}(take)];
        fprintf('appending %d points with no verified subproblem solution\n', sum(take));
    end
else
    warning('retry_log.csv not found in %s; the six unsolved points are not exported', crit_dir);
end
up = unique(probs,'stable');

alg_names = {'MPG-Explicit','MPG-Armijo','MPG-Implicit','CNPG-Max','CNPG-Avg', ...
             'EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7'};

for k = 1:numel(up)
    pname = up{k};
    rng(123); problem = pname;
    [n,m,l,u,x0_init] = inip;
    delta = (8*rand + 2)/100 * norm(x0_init);
    [dimA,A,b] = datas(n,m,delta);
    Bcell = cell(m,1);
    for i = 1:m, Bcell{i} = A{i}(1:n,:); end
    L  = load(fullfile(res_dir,[pname '_results.mat']));
    pr = L.prob_results;
    idx = find(strcmp(probs,pname));
    pts = struct('alg',{},'run',{},'x',{},'JG',{},'H',{});
    for t = 1:numel(idx)
        j = idx(t);
        a = find(strcmp(alg_names,algs{j}),1);
        X = pr{a}.x_final;
        x = X(:,runs(j));
        if any(~isfinite(x)), continue; end
        JG = zeros(m,n); H = zeros(m,1);
        for i = 1:m
            JG(i,:) = evalgradg(n,x,i);
            H(i)    = delta*norm(Bcell{i}'\x,1);
        end
        if any(~isfinite(JG(:))) || any(~isfinite(H)), continue; end
        pts(end+1).alg = algs{j};  %#ok<AGROW>
        pts(end).run   = runs(j);
        pts(end).x     = x;
        pts(end).JG    = JG;
        pts(end).H     = H;
    end
    save(fullfile(out_dir,[pname '.mat']),'pts','n','m','l','u','delta','A','b','dimA');
    fprintf('%-8s  n=%2d m=%d  %d points\n', pname, n, m, numel(pts));
end
fprintf('exported to %s\n', out_dir);
end

% ------------------------------------------------------------------------
function root = pkg_root(here)
% Folder that holds both Code/ and Code_Data/ (works from the working tree
% and from inside the released package).
d = here;
for k = 1:5
    d = fileparts(d);
    if exist(fullfile(d,'Code'),'dir') && exist(fullfile(d,'Code_Data'),'dir')
        root = d; return
    end
end
error('cannot locate the package root above %s', here);
end
