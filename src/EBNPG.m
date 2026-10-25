% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function [x,info,iter,ngev,nhev,time] = EBNPG(n,m,l,u,x,eta,sigma,gamma,beta,stop_tol,itermax,Q,A,b,dimA,mode,varargin)

% Parse the optional arguments (used by N5, N6, N7)
p = inputParser;
addOptional(p, 'M', 5);          % window size (N5, N7)
addOptional(p, 'rho', 0.9);      % forgetting factor (N7)
addOptional(p, 'theta_avg', 0.5); % averaging parameter (N6)
parse(p, varargin{:});
M = p.Results.M;
rho = p.Results.rho;
theta_avg = p.Results.theta_avg;

if nargin < 15
    mode = 'N1';
end

% If Q is not given, use the identity
if isempty(Q)
    Q = eye(n);
end

tic;

% Initialize the control vector
nu = zeros(m,1);

% Evaluate the current point
G = zeros(m,1);
H = zeros(m,1);
for i = 1:m
    G(i) = evalg(n,x,i);
    H(i) = evalh(n,x,i,A,b);
end
F = G + H;

% ---- Initialize the history data structures according to the mode ----
switch mode
    case {'N0','N1','N2','N3'}
        % No extra state needed
    case 'N5'   % Max-type
        window = zeros(m, M+1);
        window(:,1) = F;
        for j = 2:M+1
            window(:,j) = F;
        end
    case 'N6'   % Average-type (Zhang-Hager)
        Qk = 1;
        C = F;
    case 'N7'   % Forgetting-factor
        window = zeros(m, M+1);
        window(:,1) = F;
        for j = 2:M+1
            window(:,j) = F;
        end
    otherwise
        error('Unknown mode: %s', mode);
end

% Counters
ngev = 0;
nhev = 0;
iter = 0;

fprintf('----------------------------------------------------------------------\n');
fprintf('   Explicit Bregman Nonmonotone Proximal Gradient (EBNPG)           \n');
fprintf('----------------------------------------------------------------------\n');
fprintf('mode = %s, eta = %.2f, sigma = %.2f, gamma = %.2f, beta = %.2f\n', mode, eta, sigma, gamma, beta);
if ismember(mode, {'N5','N7'})
    fprintf('M = %d, rho = %.2f\n', M, rho);
elseif strcmp(mode, 'N6')
    fprintf('theta_avg = %.2f\n', theta_avg);
end
fprintf('Number of variables: %i, objectives: %i\n', n, m);
fprintf('Stop tolerance: %.0e\n', stop_tol);

fprintf('\n');
fprintf('%-5s   %-8s   %-8s   %-8s   %-10s %-10s %-8s\n', ...
        'it', '|theta|', 'alpha', '||d||', 'F1', 'F2', 'nu_norm');
fprintf('%5d   %8.2e   %8s   %8s   %+8.2e   %+8.2e   %8s\n', ...
        iter, abs(0), '-', '-', F(1), F(2), '-');

while (iter < itermax)
    iter = iter + 1;

    % 1. Compute the gradients
    JG = zeros(m,n);
    for i = 1:m
        JG(i,:) = evalgradg(n,x,i);
        ngev = ngev + 1;
    end

    % 2. Solve the Bregman proximal subproblem
    [theta, tau, p, flag] = subproblemProxGrad(n,m,l,u,x,beta,JG,H,dimA,A,b,Q);
    if flag ~= 0
        info = -2;
        time = toc;
        print_final(info, iter, ngev, nhev, time, G, H);
        return;
    end

    d = p - x;
    normd = norm(d);

    % 3. Compute Delta_i = g_i(p) - g_i(x)  (difference of the nonsmooth parts)
    delta = zeros(m,1);
    for i = 1:m
        [hp, flag_h] = evalh(n,p,i,A,b);
        nhev = nhev + 1;
        if flag_h ~= 0
            info = -1;
            time = toc;
            print_final(info, iter, ngev, nhev, time, G, H);
            return;
        end
        delta(i) = hp - H(i);
    end

    % 4. Stopping criterion
    if normd <= stop_tol
        info = 0;
        time = toc;
        print_final(info, iter, ngev, nhev, time, G, H);
        return;
    end

    % 5. Backtracking line search (uses the smooth part only)
    alpha_step = 1.0;   % initial step; may be replaced by bar_gamma_k
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
        print_final(info, iter, ngev, nhev, time, G, H);
        return;
    end

    % 6. Accept the step and update x
    x = x + alpha_step * d;

    % 7. Evaluate the smooth and nonsmooth parts at the new point
    G_new = zeros(m,1);
    H_new = zeros(m,1);
    for i = 1:m
        G_new(i) = evalg(n,x,i);
        ngev = ngev + 1;
        H_new(i) = evalh(n,x,i,A,b);
        nhev = nhev + 1;
    end
    F_new = G_new + H_new;

    % 8. Compute the endogenous relaxation contraction vector c (used by some modes)
    c = nu - (G_new - G) - alpha_step * delta;

    % 9. Update nu and the reference values according to mode
    switch mode
        case 'N0'   % monotone
            nu = zeros(m,1);
            
        case 'N1'   % diagonal endogenous transfer
            nu = (1 - eta) * c;
            nu = max(nu, 0);
            
        case 'N2'   % partial aggregation (uniform weights)
            tau_mat = 0.8;
            v = (1/m) * ones(m,1);
            P = (1-tau_mat) * eye(m) + tau_mat * (v * ones(1,m));
            nu = (1 - eta) * P * c;
            nu = max(nu, 0);
            
        case 'N3'   % inverse-contraction weighted aggregation
            tau_mat = 0.8;
            eps0 = 1e-6;
            q = 2;
            weights = (eps0 + c).^(-q);
            v = weights / sum(weights);
            P = (1-tau_mat) * eye(m) + tau_mat * (v * ones(1,m));
            nu = (1 - eta) * P * c;
            nu = max(nu, 0);
            
        case 'N5'   % Max-type (Grippo-Lampariello-Lucidi)
            % Update the window: the new value goes into column 1, the rest shift right
            window = [F_new, window(:,1:end-1)];
            R = max(window, [], 2);
            nu = R - F_new;
            
        case 'N6'   % Average-type (Zhang-Hager)
            Qk_new = 1 + theta_avg * Qk;
            C = (1 - 1/Qk_new) * C + (1/Qk_new) * F_new;
            nu = C - F_new;
            Qk = Qk_new;
            
        case 'N7'   % Forgetting-factor
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

    % 10. Print the iteration information
    if mod(iter, 10) == 0
        fprintf('\n');
        fprintf('%-5s   %-8s   %-8s   %-8s   %-10s %-10s %-8s\n', ...
                'it', '|theta|', 'alpha', '||d||', 'F1', 'F2', 'nu_norm');
    end
    fprintf('%5d   %8.2e   %8.2e   %8.2e   %+8.2e   %+8.2e   %8.2e\n', ...
            iter, abs(theta), alpha_step, normd, F_new(1), F_new(2), norm(nu,1));

    % Update the variables
    G = G_new;
    H = H_new;
    F = F_new;

end

info = 1;  % maximum number of iterations reached
time = toc;
print_final(info, iter, ngev, nhev, time, G, H);
end

% Helper: print the final result in a uniform format
% Helper: print the final result in a uniform format (any number of objectives)
function print_final(info, iter, ngev, nhev, time, G, H)
    fprintf('\n========================================\n');
    if info == 0
        fprintf('Termination status: %d (solution found)\n', info);
    elseif info == 1
        fprintf('Termination status: %d (max iterations reached)\n', info);
    elseif info == -1
        fprintf('Termination status: %d (error in H evaluation)\n', info);
    elseif info == -2
        fprintf('Termination status: %d (error in subproblem)\n', info);
    elseif info == -3
        fprintf('Termination status: %d (error in line search)\n', info);
    else
        fprintf('Termination status: %d (unknown)\n', info);
    end
    fprintf('Iterations        : %d\n', iter);
    fprintf('Grad evaluations  : %d\n', ngev);
    fprintf('H evaluations     : %d\n', nhev);
    fprintf('CPU time (s)      : %.2f\n', time);
    if nargin >= 7 && ~isempty(G) && ~isempty(H)
        % Get the number of objectives
        if length(G) == length(H)
            m_actual = length(G);
        else
            m_actual = min(length(G), length(H));
        end
        if m_actual >= 1
            fprintf('Final F           : ');
            for i = 1:min(m_actual, 5)  % print at most the first 5 objectives
                if i > 1
                    fprintf('   ');
                end
                fprintf('%+8.2e', G(i)+H(i));
            end
            if m_actual > 5
                fprintf(' ...');
            end
            fprintf('\n');
        end
    end
    fprintf('========================================\n');
end