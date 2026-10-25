% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function probe_wratio()
% Check the claim of Subsection 6.11 that, on well-conditioned problems, the
% adaptive weights approach uniform scaling: report max(w)/min(w) over the last
% five iterations of the adaptive run.
%
% Problems: three on which the adaptive kernel does NOT help (well conditioned,
% DD1 / BK1 / MGH16 are the ones where the best-variant SR drops) and NIC1,
% the ill-conditioned one, for contrast.
global problem
probs = {'DD1','BK1','MGH16','MLF2','Lov6','NIC1'};
fprintf('%-6s %10s %10s %10s\n', 'prob', 'median r', 'q25 r', 'q75 r');
for k = 1:numel(probs)
    problem = probs{k};
    rng(123);
    [n,m,l,u,x0_init] = inip;                                   %#ok<ASGLU>
    delta = (8*rand+2)/100 * norm(x0_init);
    [dimA,A,b] = datas(n,m,delta);
    R = [];
    for r = 1:5
        xr = l + (u-l).*rand(n,1);
        [~, inf, it, ~, ~, ~, wt] = EBNPG_W(n,m,l,u,xr,0.5,1e-4,0.5,1.0,1e-4, ...
            200,A,b,dimA,'N2',1e-8,'M',5,'rho',0.9,'theta_avg',0.5);
        if inf == 0 && it > 6
            R = [R, max(wt(:, it-4:it), [], 1) ./ min(wt(:, it-4:it), [], 1)]; %#ok<AGROW>
        end
    end
    if isempty(R)
        fprintf('%-6s %10s %10s %10s\n', probs{k}, '-', '-', '-');
    else
        fprintf('%-6s %10.2f %10.2f %10.2f\n', probs{k}, ...
            median(R), quantile(R, .25), quantile(R, .75));
    end
end
end
