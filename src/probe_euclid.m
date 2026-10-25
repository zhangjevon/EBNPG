% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function probe_euclid()
% Control experiment: does the present harness reproduce the STORED Euclidean
% EBNPG-N2 results of DTLZ2?
%
% In the principal experiment EBNPG-N2 is the 8th algorithm, so its 100
% starting points are draws 701-800 of the per-problem stream.
% Stored: it(1:10) = [6 55 9 10 6 40 24 29 28 19], median 17.5, SR 98.
global problem
problem = 'DTLZ2';
rng(123);
[n,m,l,u,x0_init] = inip;                                       %#ok<ASGLU>
delta = (8*rand+2)/100 * norm(x0_init);
[dimA,A,b] = datas(n,m,delta);
fprintf('n=%d m=%d delta=%.6g\n', n, m, delta);
for a = 1:700
    xr = l + (u-l).*rand(n,1);                                  %#ok<NASGU>
end
its = zeros(10,1); ngs = zeros(10,1); nhs = zeros(10,1); infs = zeros(10,1);
for r = 1:10
    xr = l + (u-l).*rand(n,1);
    [~, inf, it, ng, nh, ~] = EBNPG(n,m,l,u,xr,0.5,1e-4,0.5,1.0,1e-4,200, ...
                                    eye(n),A,b,dimA,'N2','M',5,'rho',0.9,'theta_avg',0.5);
    its(r) = it; ngs(r) = ng; nhs(r) = nh; infs(r) = inf;
end
fprintf('EUCLID  it(1:10) = %s\n', mat2str(its'));
fprintf('EUCLID  ng(1:10) = %s\n', mat2str(ngs'));
fprintf('EUCLID  nh(1:10) = %s\n', mat2str(nhs'));
fprintf('STORED  it(1:10) = [6 55 9 10 6 40 24 29 28 19]\n');
end
