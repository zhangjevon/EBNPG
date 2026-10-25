% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
% R2 revision: fully documented rerun of MGH33 for all twelve algorithms.
% Protocol = the one documented in Subsection 6.2 of the manuscript:
%   rng(123); [n,m,l,u,x_ref] = inip; delta = (8*rand+2)/100*norm(x_ref);
%   [dimA,A,b] = datas(n,m,delta); twelve algorithm slots x 100 starts.
addpath(fileparts(mfilename('fullpath')));
if ~exist('results','dir'), mkdir('results'); end
fprintf('=== R2 rerun MGH33 start ===\n');
run_problem('MGH33', 100);
fprintf('=== R2 rerun MGH33 done ===\n');
