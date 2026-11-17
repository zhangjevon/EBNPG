% Added for the EBNPG manuscript (third revision).
% SPDX-License-Identifier: GPL-3.0-or-later
function cert_sample(prob_list, stride, out_csv)
% CERT_SAMPLE  Independent re-certification of a sample of the first-pass
% values stored in Code_Data/crit_out.
%
% The first pass solved the common proximal subproblem at 49,793 of the
% 50,400 returned points with one fixed setting.  Those values are used in
% Table 4 as they stand, so this script re-solves a systematic sample
% (every <stride>-th point) with the *same* setting as the first pass and
% reports
%   d_theta = theta_recomputed - theta_stored
%   gap     = Theta(model at the returned point) - theta_recomputed
% A value is certified when |gap| <= max(1e-10, 1e-12|theta|).
%
% Usage: cert_sample(prob_list, 25, 'cert_sample.csv')

if nargin < 2 || isempty(stride),   stride = 25; end
if nargin < 3, out_csv = 'cert_sample.csv'; end
root     = pkg_root(mfilename('fullpath'));   % folder holding Code/ and Code_Data/
code_dir = fullfile(root,'Code');
data_dir = fullfile(root,'Code_Data');
old_pwd = pwd; cd(data_dir); cleanup = onCleanup(@() cd(old_pwd)); %#ok<NASGU>
addpath(code_dir);
global problem

fid = fopen(out_csv,'w');
fprintf(fid,'prob,alg,run,theta_stored,theta_recomputed,d_theta,gap,dnorm,cert\n');
cnt = 0; nbad = 0; ndiff = 0; worst = 0;
for pi = 1:numel(prob_list)
    p = prob_list{pi};
    rng(123); problem = p;
    [n,m,l,u,x0_init] = inip;
    delta = (8*rand + 2)/100 * norm(x0_init);
    [dimA,A,b] = datas(n,m,delta);
    Bc = cell(m,1); for i = 1:m, Bc{i} = A{i}(1:n,:); end
    pr = load(fullfile('results',[p '_results.mat'])).prob_results;
    o  = load(fullfile('crit_out',[p '_crit.mat'])).out;
    k = 0;
    for a = 1:numel(pr)
        r = pr{a}; if isempty(r), continue; end
        for run = 1:size(r.x_final,2)
            k = k + 1;
            if mod(k,stride) ~= 0, continue; end
            if ~isfinite(o.crit(a,run)), continue; end
            x = r.x_final(:,run); if any(~isfinite(x)), continue; end
            JG = zeros(m,n); H = zeros(m,1);
            for i = 1:m
                JG(i,:) = evalgradg(n,x,i);
                H(i)    = delta*norm(Bc{i}'\x,1);
            end
            [theta,~,pp,flag] = subproblemProxGrad(n,m,l,u,x,1,JG,H,dimA,A,b,eye(n));
            if flag ~= 0 || ~isfinite(theta), continue; end
            d = pp - x;
            cd_ = zeros(m,1);
            for i = 1:m
                cd_(i) = JG(i,:)*d + delta*norm(Bc{i}'\pp,1) - H(i);
            end
            gap = max(cd_) + 0.5*(d'*d) - theta;
            tol = max(1e-10, 1e-12*abs(theta));
            ok  = abs(gap) <= tol;
            cnt = cnt + 1; nbad = nbad + (~ok);
            dth = theta + o.crit(a,run);            % stored crit = -theta
            if abs(dth) > 1e-8, ndiff = ndiff + 1; end
            worst = max(worst, abs(dth));
            fprintf(fid,'%s,%s,%d,%.12g,%.12g,%.6g,%.6g,%.12g,%d\n', ...
                    p,r.algorithm,run,-o.crit(a,run),-theta,-dth,gap,norm(d),ok);
        end
    end
    fprintf('%-6s sampled so far %d\n', p, cnt);
end
fclose(fid);
fprintf('\nsampled            : %d\n', cnt);
fprintf('certified          : %d  (%.2f%%)\n', cnt-nbad, 100*(cnt-nbad)/max(cnt,1));
fprintf('|d_theta| > 1e-8   : %d   max=%.3e\n', ndiff, worst);
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
