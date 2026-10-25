% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function [x,info,iter,ngev,nhev,time,wtrace] = EBNPG_W(n,m,l,u,x,eta,sigma,gamma,beta,stop_tol,itermax,A,b,dimA,mode,epsx,varargin)
% The optional seventh output wtrace collects the weight vector of every
% iteration (n x iter).  It is only used for diagnostics; forming it does not
% change the iteration.
% EBNPG_W -- EBNPG with the adaptive BB-type diagonal variable metric of
% Subsection 6.11 (EBNPG-W).
%
% Identical to EBNPG.m except that the fixed kernel matrix Q is replaced, at
% every iteration, by W^k = diag(w^k) built from the coordinatewise BB
% curvature estimate of the two most recent iterates.  The update reuses the
% gradients already computed for the Jacobian, so it adds no gradient
% evaluations (ngev counts are unchanged).
%
% Weight rule (manuscript, Subsection 6.11):
%   kappa_ij^k = | grad f_i(x^k)_j - grad f_i(x^{k-1})_j |
%                / max{ |x^k_j - x^{k-1}_j|, epsx }
%   raw_j      = max_i kappa_ij^k
%   w <- 0.5 * raw + 0.5 * w            (blending parameter 0.5)
%   w <- w / geomean(w)                 (geometric-mean normalisation)
%   w <- min(max(w, 1e-4), 1e2)         (safeguard)
% Iteration 1 has no previous iterate, so W^1 = I.

p = inputParser;
addOptional(p, 'M', 5);
addOptional(p, 'rho', 0.9);
addOptional(p, 'theta_avg', 0.5);
addOptional(p, 'Scheme', 'base');   % candidate weight-update variants
parse(p, varargin{:});
M = p.Results.M;
rho = p.Results.rho;
theta_avg = p.Results.theta_avg;
scheme = p.Results.Scheme;

if nargin < 14
    mode = 'N1';
end
if nargin < 15 || isempty(epsx)
    epsx = 1e-8;
end

WLOW  = 1e-4;
WUP   = 1e2;
BLEND = 0.5;

tic;

nu = zeros(m,1);

G = zeros(m,1);
H = zeros(m,1);
for i = 1:m
    G(i) = evalg(n,x,i);
    H(i) = evalh(n,x,i,A,b);
end
F = G + H;

switch mode
    case {'N0','N1','N2','N3'}
    case 'N5'
        window = zeros(m, M+1);
        window(:,1) = F;
        for j = 2:M+1
            window(:,j) = F;
        end
    case 'N6'
        Qk = 1;
        C = F;
    case 'N7'
        window = zeros(m, M+1);
        window(:,1) = F;
        for j = 2:M+1
            window(:,j) = F;
        end
    otherwise
        error('Unknown mode: %s', mode);
end

ngev = 0;
nhev = 0;
iter = 0;

% ---- adaptive diagonal weight state ----
w     = ones(n,1);
JGold = [];
xold  = [];
WTRACE = zeros(n, itermax);

while (iter < itermax)
    iter = iter + 1;

    % 1. gradients at the current iterate (reused by the weight update)
    JG = zeros(m,n);
    for i = 1:m
        JG(i,:) = evalgradg(n,x,i);
        ngev = ngev + 1;
    end

    % 1b. adaptive BB-type diagonal kernel: use the two most recent iterates
    if iter >= 2
        kap = abs(JG - JGold);                          % m x n
        dx  = max(abs(x - xold), epsx);                 % n x 1
        kd  = kap ./ repmat(dx', m, 1);                 % curvature per objective
        raw = (max(kd,[],1))';                          % n x 1 : max over objectives
        switch scheme
        case 'base'                                     % manuscript reading
            w = BLEND*raw + (1-BLEND)*w;
            w = gnorm(w);   w = min(max(w,WLOW),WUP);
        case 'nonorm'                                   % no normalisation
            w = BLEND*raw + (1-BLEND)*w;
            w = min(max(w,WLOW),WUP);
        case 'normfirst'                                % normalise raw, then blend
            r = gnorm(raw);
            w = BLEND*r + (1-BLEND)*w;
            w = gnorm(w);   w = min(max(w,WLOW),WUP);
        case 'amean'                                    % arithmetic-mean normalisation
            w = BLEND*raw + (1-BLEND)*w;
            w = w / mean(w); w = min(max(w,WLOW),WUP);
        case 'noblend'                                  % no smoothing
            w = gnorm(raw); w = min(max(w,WLOW),WUP);
        case 'meanobj'                                  % mean over objectives
            raw = (mean(kd,1))';
            w = BLEND*raw + (1-BLEND)*w;
            w = gnorm(w);   w = min(max(w,WLOW),WUP);
        case 'sumobj'                                   % gradient of the sum
            kd = abs(sum(JG - JGold, 1))' ./ dx;
            w = BLEND*kd + (1-BLEND)*w;
            w = gnorm(w);   w = min(max(w,WLOW),WUP);
        case 'geoblend'                                 % geometric blending 0.5
            w = sqrt(max(raw,realmin) .* max(w,realmin));
            w = gnorm(w);   w = min(max(w,WLOW),WUP);
        case 'clipthennorm'                             % clip, then normalise
            w = BLEND*raw + (1-BLEND)*w;
            w = min(max(w,WLOW),WUP); w = gnorm(w);
        case 'blendfirst'                               % smooth kappa, then max
            w = BLEND*raw + (1-BLEND)*w;
            w = gnorm(w);   w = min(max(w,WLOW),WUP);
        otherwise
            error('Unknown scheme: %s', scheme);
        end
    end
    Q = diag(w);
    WTRACE(:, iter) = w;

    % remember this iterate for the next weight update (before x moves)
    JGold = JG;
    xold  = x;

    % 2. Bregman proximal subproblem with the current kernel
    [theta, tau, p, flag] = subproblemProxGrad(n,m,l,u,x,beta,JG,H,dimA,A,b,Q);
    if flag ~= 0
        info = -2;
        time = toc;
        wtrace = WTRACE(:, 1:max(iter,1));
        return;
    end

    d = p - x;
    normd = norm(d);

    % 3. g_i(p) - g_i(x)
    delta = zeros(m,1);
    for i = 1:m
        [hp, flag_h] = evalh(n,p,i,A,b);
        nhev = nhev + 1;
        if flag_h ~= 0
            info = -1;
            time = toc;
            wtrace = WTRACE(:, 1:max(iter,1));
            return;
        end
        delta(i) = hp - H(i);
    end

    % 4. stopping rule
    if normd <= stop_tol
        info = 0;
        time = toc;
        wtrace = WTRACE(:, 1:max(iter,1));
        return;
    end

    % 5. backtracking on the smooth part
    alpha_step = 1.0;
    max_backtrack = 50;
    bt = 0;
    while bt < max_backtrack
        xtrial = x + alpha_step * d;
        Gtrial = zeros(m,1);
        accepted = true;
        for i = 1:m
            Gtrial(i) = evalg(n,xtrial,i);
            ngev = ngev + 1;
            if Gtrial(i) > G(i) + alpha_step*(sigma*theta - delta(i)) + nu(i)
                accepted = false;
                break;
            end
        end
        if accepted
            break;
        end
        alpha_step = gamma * alpha_step;
        bt = bt + 1;
    end
    if bt == max_backtrack
        info = -3;
        time = toc;
        wtrace = WTRACE(:, 1:max(iter,1));
        return;
    end

    % 6. accept
    x = x + alpha_step * d;

    % 7. re-evaluate
    G_new = zeros(m,1);
    H_new = zeros(m,1);
    for i = 1:m
        G_new(i) = evalg(n,x,i);
        ngev = ngev + 1;
        H_new(i) = evalh(n,x,i,A,b);
        nhev = nhev + 1;
    end
    F_new = G_new + H_new;

    % 8. endogenous slack
    c = nu - (G_new - G) - alpha_step * delta;

    % 9. nonmonotonicity control
    switch mode
        case 'N0'
            nu = zeros(m,1);
        case 'N1'
            nu = (1 - eta) * c;
            nu = max(nu, 0);
        case 'N2'
            tau_mat = 0.8;
            v = (1/m) * ones(m,1);
            P = (1-tau_mat) * eye(m) + tau_mat * (v * ones(1,m));
            nu = (1 - eta) * P * c;
            nu = max(nu, 0);
        case 'N3'
            tau_mat = 0.8;
            eps0 = 1e-6;
            q = 2;
            weights = (eps0 + c).^(-q);
            v = weights / sum(weights);
            P = (1-tau_mat) * eye(m) + tau_mat * (v * ones(1,m));
            nu = (1 - eta) * P * c;
            nu = max(nu, 0);
        case 'N5'
            window = [F_new, window(:,1:end-1)];
            R = max(window, [], 2);
            nu = R - F_new;
        case 'N6'
            Qk_new = 1 + theta_avg * Qk;
            C = (1 - 1/Qk_new) * C + (1/Qk_new) * F_new;
            nu = C - F_new;
            Qk = Qk_new;
        case 'N7'
            window = [F_new, window(:,1:end-1)];
            R = zeros(m,1);
            for i = 1:m
                vals = zeros(M+1,1);
                for j = 0:M
                    vals(j+1) = rho^j * window(i, j+1);
                end
                R(i) = max(vals);
            end
            nu = R - F_new;
        otherwise
            error('Unknown mode: %s', mode);
    end

    G = G_new;
    H = H_new;
    F = F_new;

    if (min(w) <= 0) || ~all(isfinite(w))
        w = ones(n,1);
    end
end

info = 1;
time = toc;
wtrace = WTRACE(:, 1:max(iter,1));
end

function w = gnorm(w)
% divide by the geometric mean
g = exp(mean(log(max(w, realmin))));
if (g > 0) && isfinite(g)
    w = w ./ g;
else
    w = ones(size(w));
end
end
