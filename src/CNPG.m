% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function [x, info, iter, ngev, nhev, time] = CNPG(n, m, l, u, x, sigma, gamma, stop_tol, itermax, mode, A, b, dimA, varargin)
% CNPG - Composite Nonmonotone Proximal-Gradient Method
%   for Composite Multiobjective Optimization
%
%   This implements the composite nonmonotone proximal-gradient method
%   described in [Mita-Fukuda-Yamashita, JOTA 2025], which evaluates the
%   FULL composite objective F_i = g_i + h_i at trial points during the
%   nonmonotone line search (as opposed to EBNPG which uses only the
%   smooth part g_i in its line search test).
%
%   Two nonmonotone reference strategies:
%     'max'  - Max-type (Grippo-Lampariello-Lucidi) reference
%     'avg'  - Average-type (Zhang-Hager) reference
%
%   Key distinction from EBNPG:
%     EBNPG : Bregman distance (kernel Q) + smooth-part line search
%     CNPG  : Euclidean distance (Q = I)  + full-composite line search

    p = inputParser;
    addOptional(p, 'M', 5);             % Window size (max-type)
    addOptional(p, 'theta_avg', 0.5);   % Average-type parameter
    parse(p, varargin{:});
    M = p.Results.M;
    theta_avg = p.Results.theta_avg;

    if nargin < 10
        mode = 'max';
    end

    tic;

    % Counters
    ngev = 0;
    nhev = 0;
    iter = 0;

    % Evaluate initial point
    G = zeros(m, 1);
    H = zeros(m, 1);
    for i = 1:m
        G(i) = evalg(n, x, i);
        ngev = ngev + 1;
        [H(i), flag] = evalh(n, x, i, A, b);
        nhev = nhev + 1;
        if flag ~= 0
            info = -1;
            time = toc;
            return;
        end
    end
    F = G + H;  % Full composite objective

    % Initialize reference data structure
    switch mode
        case 'max'
            % Max-type: sliding window of last M+1 composite values
            window = zeros(m, M + 1);
            for j = 1:M+1
                window(:, j) = F;
            end
        case 'avg'
            % Average-type (Zhang-Hager): running weighted average
            Qk = 1;
            C = F;
    end

    while iter < itermax
        iter = iter + 1;

        % 1. Compute Jacobian of smooth part G
        JG = zeros(m, n);
        for i = 1:m
            JG(i, :) = evalgradg(n, x, i);
            ngev = ngev + 1;
        end

        % 2. Solve Euclidean proximal subproblem (Q = I, alpha = 1)
        [theta, tau, p, flag] = subproblemProxGrad(n, m, l, u, x, 1, JG, H, dimA, A, b, []);
        if flag ~= 0
            info = -2;
            time = toc;
            return;
        end

        d = p - x;
        normd = norm(d);

        % 3. Stopping criterion
        if normd <= stop_tol
            info = 0;
            time = toc;
            return;
        end

        % 4. Nonmonotone line search on FULL COMPOSITE objective
        alpha_step = 1.0;
        max_backtrack = 50;
        bt = 0;

        % Compute reference values R_i
        switch mode
            case 'max'
                R = max(window, [], 2);
            case 'avg'
                R = C;
        end

        while bt < max_backtrack
            xtrial = x + alpha_step * d;

            % Evaluate full composite F at trial point
            Ftrial = zeros(m, 1);
            accepted = true;
            for i = 1:m
                gi = evalg(n, xtrial, i);
                ngev = ngev + 1;
                [hi, flag_h] = evalh(n, xtrial, i, A, b);
                nhev = nhev + 1;
                if flag_h ~= 0
                    info = -1;
                    time = toc;
                    return;
                end
                Ftrial(i) = gi + hi;

                % Nonmonotone acceptance test:
                %   F_i(x + alpha*d) <= R_i - sigma * alpha * |theta|
                if Ftrial(i) > R(i) - sigma * alpha_step * abs(theta)
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
            return;
        end

        % 5. Accept step and update
        x = x + alpha_step * d;

        % Evaluate new point
        G_new = zeros(m, 1);
        H_new = zeros(m, 1);
        for i = 1:m
            G_new(i) = evalg(n, x, i);
            ngev = ngev + 1;
            [H_new(i), flag] = evalh(n, x, i, A, b);
            nhev = nhev + 1;
            if flag ~= 0
                info = -1;
                time = toc;
                return;
            end
        end
        F_new = G_new + H_new;

        % 6. Update reference data structure
        switch mode
            case 'max'
                % Slide window: newest value enters from left
                window = [F_new, window(:, 1:end-1)];

            case 'avg'
                % Zhang-Hager update: Q_{k+1} = 1 + theta*Q_k
                Qk_new = 1 + theta_avg * Qk;
                C = (1 - 1/Qk_new) * C + (1/Qk_new) * F_new;
                Qk = Qk_new;
        end

        F = F_new;
        G = G_new;
        H = H_new;
    end

    info = 1;  % max iterations reached
    time = toc;
end
