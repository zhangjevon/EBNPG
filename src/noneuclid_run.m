% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
% Fixed non-Euclidean kernel experiment (referee Comment 2).
%
% For each of the 42 problems, EBNPG-N2 is run twice, from the SAME 100
% starting points, with
%   K0 : omega(x) = 1/2 ||x||^2            (Euclidean, M = I)
%   K1 : omega(x) = 1/2 x' diag(w) x       (fixed diagonal kernel, w frozen)
% so the comparison is paired run by run.
%
% The weight vector w is built ONCE per problem, before any run, from an
% averaged finite-difference estimate of the coordinate curvature of the smooth
% components over the box; it is then held fixed for all iterations and all
% starting points, so the kernel satisfies the fixed-kernel theory.
%
% Uses the ORIGINAL EBNPG (linprog evaluation of g_i), so that K0 reproduces
% the stored EBNPG-N2 results exactly on the 41 problems whose fixed-kernel
% study shares their starting points (the exceptions are AP4 and DGO2; see
% the block map below).  Results are saved per problem so that the run can
% be resumed.
function noneuclid_run()
    probs = {'AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS', ...
             'Hil1','IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26', ...
             'MGH33','MLF2','MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1', ...
             'NIC2','NIC3','NIM1','NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4', ...
             'Toi9','VU2','ZDT6','ZLT1'};
    MODES  = {'N2'};
    outdir = 'ne_full';
    if ~exist(outdir,'dir'), mkdir(outdir); end

    for pi = 1:numel(probs)
        pname = probs{pi};
        for mi = 1:numel(MODES)
            mode = MODES{mi};
            outfile = fullfile(outdir,[pname '_' mode '.mat']);
            if exist(outfile,'file')
                fprintf('%-6s %-3s already done, skipping\n', pname, mode);
                continue
            end
            clear global problem
            global problem
            problem = pname;
            rng(123);
            [n,m,l,u,x0_init] = inip;                 %#ok<ASGLU>
            delta = (8*rand+2)/100 * norm(x0_init);
            [dimA,A,b] = datas(n,m,delta);
            % Starting points of the paired runs: the 8th block of 100 draws
            % for 40 of the 42 problems, the 7th for BK1 and PNR.  This
            % reproduces the stored fixed-kernel study exactly.  It is also
            % the EBNPG-N2 block of the main experiments for 41 problems;
            % the exceptions are AP4 (N2 used the 7th block) and DGO2 (the
            % 5th), whose sessions were interrupted and resumed without
            % advancing the stream, while their fixed-kernel study was
            % generated from the 8th block.
            NSKIP = 7;
            if ismember(pname, {'BK1','PNR'})
                NSKIP = 6;
            end
            for a = 1:NSKIP
                for r = 1:100
                    x0r = l + (u-l).*rand(n,1);       %#ok<NASGU>
                end
            end
            S = zeros(n,100);                          % shared starting points
            for r = 1:100
                S(:,r) = l + (u-l).*rand(n,1);
            end

            w  = curvature_weights(n,m,l,u);
            M1 = diag(w);

            R = struct();
            t0 = tic;
            for kk = 1:2
                if kk == 1, Qk = eye(n); nm = 'K0';
                else,       Qk = M1;     nm = 'K1'; end
                iter = zeros(100,1); info = zeros(100,1);
                ngev = zeros(100,1); nhev = zeros(100,1);
                X    = zeros(n,100);
                [~, iter, info, ngev, nhev, X] = evalc( ...
                    'batch_ebnpg_fast(n,m,l,u,S,Qk,A,b,dimA,mode)');
                R.(nm).iter = iter; R.(nm).info = info;
                R.(nm).ngev = ngev; R.(nm).nhev = nhev;
                R.(nm).X    = X;
            end
            R.w = w;
            save(outfile,'R');
            sr0 = 100*sum(R.K0.info == 0)/100;
            sr1 = 100*sum(R.K1.info == 0)/100;
            fprintf('%-6s %-3s  w in [%.3g, %.3g]  SR: K0=%5.1f  K1=%5.1f   (%5.1fs)\n', ...
                    pname, mode, min(w), max(w), sr0, sr1, toc(t0));
        end
    end
    fprintf('\nALL DONE\n');
end

function [iter, info, ngev, nhev, X] = batch_ebnpg_fast(n,m,l,u,S,Q,A,b,dimA,mode)
    iter = zeros(100,1); info = zeros(100,1);
    ngev = zeros(100,1); nhev = zeros(100,1);
    X    = zeros(n,100);
    for r = 1:100
        [X(:,r), info(r), iter(r), ngev(r), nhev(r), ~] = EBNPG( ...
            n,m,l,u,S(:,r),0.5,1e-4,0.5,1.0,1e-4,200,Q,A,b,dimA,mode, ...
            'M',5,'rho',0.9,'theta_avg',0.5);
    end
end

function w = curvature_weights(n,m,l,u)
    % Fixed diagonal weights: averaged |second difference| of the smooth
    % components over K sample points in the box, normalised to mean 1.
    K = 20;
    rng(7);                                  % independent of the run stream
    P = l + (u-l).*rand(n,K);
    d = zeros(n,1);
    hscale = 1e-3;
    for k = 1:K
        for j = 1:n
            acc = 0;
            for i = 1:m
                xk = P(:,k);
                hj = hscale * max(1, abs(xk(j)));
                xp = xk; xm = xk;
                xp(j) = min(u(j), xk(j) + hj);
                xm(j) = max(l(j), xk(j) - hj);
                if xp(j) - xm(j) < 1e-12, continue, end
                f0 = evalg(n, xk, i);
                fp = evalg(n, xp, i);
                fm = evalg(n, xm, i);
                hj2 = xp(j) - xm(j);
                acc = acc + abs((fp - 2*f0 + fm) / (hj2*hj2/4));
            end
            d(j) = d(j) + acc / m;
        end
    end
    d = d / K;
    d = max(d, 1e-10);
    w = d / mean(d);
    w = min(max(w, 1e-2), 1e2);
end
