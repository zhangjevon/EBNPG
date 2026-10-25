% Modified from the MPG repository (https://github.com/lfprudente/MPG); see CHANGES.md.
% SPDX-License-Identifier: GPL-3.0-or-later
function [h,info] = evalh(n,x,ind,A,b)

% Use optimoptions to set the effective algorithm
options = optimoptions('linprog', ...
    'Algorithm', 'dual-simplex', ...
    'Display', 'off', ...
    'OptimalityTolerance', 1e-10);   % set to 1e-10

[xopt,fmin,flag] = linprog(-x, A{ind}, b{ind}, [], [], -inf(n,1), inf(n,1), options);

if ( flag == 1 )
    info = 0;
    h = -fmin;
else
    % Optionally print debugging information
    % fprintf('linprog flag = %d\n', flag);
    info = -1;
    h = NaN;
end