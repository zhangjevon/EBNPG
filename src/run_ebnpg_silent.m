% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function [x, info, iter, ngev, nhev, time] = run_ebnpg_silent(n,m,l,u,x,eta,sigma,gamma,beta,stop_tol,itermax,Q,A,b,dimA,mode,M,rho,theta_avg)
% Wrapper for EBNPG that suppresses verbose output via evalc
[~, x, info, iter, ngev, nhev, time] = evalc(...
    'EBNPG(n,m,l,u,x,eta,sigma,gamma,beta,stop_tol,itermax,Q,A,b,dimA,mode,''M'',M,''rho'',rho,''theta_avg'',theta_avg)');
end
