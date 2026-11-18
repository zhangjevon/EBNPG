function adjudicate32(out_csv)
% LEGACY diagnostic only; enters no final table. Use certify_adjudicated and
% the exact-rational Python certificate/replay for the final adjudication.
% ADJUDICATE32  Decide the verdict of the returned points at which the R4 pass
% could not certify the value of the common proximal subproblem.
%
% Scope.  Of the 134 points without a certified value, 128 carry a value and 6
% carry none.  Of the 128, 96 have |Theta| >= 1e3, so their 1e-4 verdict cannot
% be changed by any numerical error of the size observed here; they stay
% failures.  This script (re)examines the 32 points with |Theta| <= 1e-2 and
% the 6 unsolved ones, i.e. every point whose verdict is actually in doubt.
%
% Method.  The model is  Theta(x,d) = max_i{ grad f_i(x)'d + h_i(x+d) - h_i(x) }
% + 1/2||d||^2  on the box x+d in [l,u].  For a returned direction d the value
% Theta(x,d) is a feasible UPPER bound for the optimal value; a certificate for
% the criterion needs the sign of the optimal value, so the point is accepted
% only through the same rule as the R4 pass: the solver value must be <= 0 and
% the model evaluated at the point the solver returns must reproduce it to
% max(1e-6, 1e-12|value|).  What is added here is a wider ladder (more stages,
% a second algorithm of quadprog, tighter tolerances, two starting points and
% two epigraph scalings), because the failure of the R4 pass at these points
% was a sign artefact of the value (it came back positive by ~1e-9, which is
% impossible for a model whose value at d = 0 is 0), not a genuine
% disagreement between solver and model.
%
% OUTPUT  <out_csv>: prob,alg,run,crit_old,dn_old,theta_qp,theta_def,gap,dn,cert,stage
%
% Usage:  adjudicate32                      (writes adj32.csv next to this file)
%         adjudicate32('adj32.csv')

if nargin < 1 || isempty(out_csv)
    out_csv = fullfile(fileparts(mfilename('fullpath')),'adj32.csv');
end
[~, legacy_name, legacy_extension] = fileparts(out_csv);
if any(strcmpi([legacy_name legacy_extension], ...
        {'adjudicated.csv','adjudication_witnesses.json'}))
    error('adjudicate32:legacyOutput', ...
        'Legacy diagnostics cannot overwrite the final adjudication files.');
end

% ---- locate the package root (works in the repository and in the release) --
root = fileparts(mfilename('fullpath'));
while ~(exist(fullfile(root,'Code'),'dir') && exist(fullfile(root,'Code_Data'),'dir'))
    up = fileparts(root);
    if isempty(up) || strcmp(up,root), error('adjudicate32:root','package root not found'); end
    root = up;
end
code_dir = fullfile(root,'Code');
data_dir = fullfile(root,'Code_Data');
crit_dir = fullfile(data_dir,'crit_out_r4');
res_dir  = fullfile(data_dir,'results');

old_pwd = pwd; cd(data_dir); cleanup = onCleanup(@() cd(old_pwd)); %#ok<NASGU>
addpath(code_dir);
global problem

probs = {'AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1', ...
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2', ...
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1', ...
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1'};

cert_tol = 1e-6;
fid = fopen(out_csv,'w');
fprintf(fid,'prob,alg,run,crit_old,dn_old,theta_qp,theta_def,gap,dn,cert,stage\n');

nsel = 0; ncert = 0;
for pi = 1:numel(probs)
    p = probs{pi};
    rng(123); problem = p;
    [n,m,l,u,x0_init] = inip;
    delta = (8*rand + 2)/100 * norm(x0_init);
    [dimA,A,b] = datas(n,m,delta);
    Bc = cell(m,1); for i = 1:m, Bc{i} = A{i}(1:n,:); end

    pr = load(fullfile(res_dir,[p '_results.mat'])).prob_results;
    o  = load(fullfile(crit_dir,[p '_crit.mat'])).out;

    for a = 1:numel(pr)
        r = pr{a}; if isempty(r), continue; end
        for run = 1:size(r.x_final,2)
            cold = o.crit(a,run); dold = o.dnorm(a,run);
            isdoubt = (~isfinite(cold)) || (isfinite(cold) && abs(cold) <= 1e-2);
            if ~isdoubt, continue; end
            if isfinite(cold) && o.cert(a,run) > 0, continue; end   % already certified
            nsel = nsel + 1;

            x = r.x_final(:,run);
            if any(~isfinite(x))
                fprintf(fid,'%s,%s,%d,%g,%g,NaN,NaN,NaN,NaN,0,-1\n',p,r.algorithm,run,cold,dold);
                continue
            end
            JG = zeros(m,n); H = zeros(m,1);
            for i = 1:m
                JG(i,:) = evalgradg(n,x,i);
                H(i)    = delta*norm(Bc{i}'\x,1);
            end
            if any(~isfinite(JG(:))) || any(~isfinite(H))
                fprintf(fid,'%s,%s,%d,%g,%g,NaN,NaN,NaN,NaN,0,-1\n',p,r.algorithm,run,cold,dold);
                continue
            end

            [th_qp,th_def,gap,dn,cert,stage] = wide_ladder(n,m,l,u,x,JG,H,dimA,A,b,Bc,delta,cert_tol);
            ncert = ncert + double(cert);
            fprintf(fid,'%s,%s,%d,%.12g,%.12g,%.12g,%.12g,%.6g,%.12g,%d,%d\n', ...
                    p,r.algorithm,run,cold,dold,th_qp,th_def,gap,dn,cert,stage);
        end
    end
    fprintf('%-6s examined so far %3d, certified %3d\n',p,nsel,ncert);
end
fclose(fid);
fprintf('\nexamined : %d\ncertified: %d\ncsv      : %s\n',nsel,ncert,out_csv);
end

% ------------------------------------------------------------------------
function [best_qp,best_def,best_gap,best_dn,best_cert,best_stage] = ...
        wide_ladder(n,m,l,u,x,JG,H,dimA,A,b,Bcell,delta,cert_tol)
% Order of preference: a certified value (stage value <= 0 AND reproduced by
% the model) with the smallest |gap|; otherwise the candidate with the smallest
% |gap| among the stages that converged inside the box.
best_qp = NaN; best_def = NaN; best_gap = NaN; best_dn = NaN;
best_cert = 0; best_stage = -1;
bg = inf;
s = max([1, max(abs(JG(:))), max(abs(H))]);
starts = {[0;x], [-1e-8;x]};
stages = {};
k = 0;
for alg = {'interior-point-convex','active-set'}
    for sc = [1, s]
        for tol = [1e-10, 1e-12, 1e-13]
            for st = 1:numel(starts)
                k = k + 1;
                stages{k} = {alg{1}, sc, qopt(tol, 50000), starts{st}}; %#ok<AGROW>
            end
        end
    end
end

for k = 1:numel(stages)
    c = stages{k};
    [xo,info] = qp_scaled(n,m,l,u,x,JG,H,dimA,A,b,c{2},c{3},c{1},c{4});
    if info ~= 1 || any(~isfinite(xo)), continue; end
    pnt = xo(2:n+1); d = pnt - x;
    if ~(all(pnt >= l - 1e-9) && all(pnt <= u + 1e-9)), continue; end
    th_qp  = c{2}*xo(1) + 0.5*(d'*d);
    th_def = primal_theta(n,m,d,pnt,x,JG,H,Bcell,delta);
    g      = th_def - th_qp;
    tol    = max(cert_tol, 1e-12*abs(th_qp));
    certified = (abs(g) <= tol) && (th_qp <= 0);
    if certified
        if best_cert == 0 || abs(g) < bg
            best_qp = th_qp; best_def = th_def; best_gap = g; best_dn = norm(d);
            best_cert = 1; best_stage = k; bg = abs(g);
        end
    elseif best_cert == 0 && abs(g) < bg
        best_qp = th_qp; best_def = th_def; best_gap = g; best_dn = norm(d);
        best_stage = k; bg = abs(g);
    end
end
end

% ------------------------------------------------------------------------
function th = primal_theta(n,m,d,p,x,JG,H,Bcell,delta) %#ok<INUSD>
c = zeros(m,1);
for i = 1:m
    hp = delta*norm(Bcell{i}'\p, 1);
    c(i) = JG(i,:)*d + hp - H(i);
end
th = max(c) + 0.5*(d'*d);
end

% ------------------------------------------------------------------------
function opt = qopt(otol, maxit)
opt = optimoptions('quadprog','Display','off', ...
    'OptimalityTolerance',otol,'ConstraintTolerance',max(otol,1e-12), ...
    'StepTolerance',1e-14,'MaxIterations',maxit);
end

% ------------------------------------------------------------------------
function [xopt,info] = qp_scaled(n,m,l,u,x,JG,H,dimA,A,b,s,opt,alg,x0)
if nargin < 12 || isempty(alg), alg = 'interior-point-convex'; end
opt = optimoptions(opt,'Algorithm',alg);
sdimA = sum(dimA);
Hess  = zeros(1+n+sdimA); Hess(2:n+1,2:n+1) = eye(n);
c     = zeros(1+n+sdimA,1); c(1) = s; c(2:n+1) = -x;
index(1) = n+1;
for ind = 2:m+1, index(ind) = index(ind-1) + dimA(ind-1); end
Asub = zeros(m,1+n+sdimA); Aeq = zeros(n*m,1+n+sdimA);
for ind = 1:m
    Asub(ind,1:1+n) = [-1 JG(ind,:)/s];
    Asub(ind,index(ind)+1:index(ind+1)) = b{ind}/s;
    Aeq((ind-1)*n+1:ind*n,2:n+1) = -eye(n);
    Aeq((ind-1)*n+1:ind*n,index(ind)+1:index(ind+1)) = A{ind}';
end
bsub = zeros(m,1);
for ind = 1:m, bsub(ind) = (H(ind) + JG(ind,:)*x)/s; end
beq = zeros(n*m,1);
lb = [-inf; l; zeros(sdimA,1)]; ub = [inf; u; inf(sdimA,1)];
try
    [xopt,~,info] = quadprog(Hess,c,Asub,bsub,Aeq,beq,lb,ub,x0,opt);
catch
    xopt = NaN; info = -99;
end
end
