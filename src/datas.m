% Modified from the MPG repository (https://github.com/lfprudente/MPG); see CHANGES.md.
% SPDX-License-Identifier: GPL-3.0-or-later
function [dimA,A,b] = datas(n,m,delta)

dimA(1:m) = 2*n;

l = -10;
u =  10;

for ind = 1:m
    B = l + rand(n) * (u-l);
    A{ind} = [B; -B];
    b{ind}   = delta * ones(2*n,1);
end