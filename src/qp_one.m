% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function qp_one(prob_name)
% Count the proximal (QP) solves of MPG-Implicit for one problem.
    addpath(fileparts(mfilename('fullpath')));
    if ~exist('results','dir'), mkdir('results'); end
    global problem QPSOLVECOUNT
    problem = prob_name;

    R = load(fullfile('results', [prob_name '_results.mat']));
    it_ref = R.prob_results{3}.iter_vals(1:8);

    offset = -1;
    for skip = [2 1 0 3]
        rng(123);
        [n, m, l, u, x0_init] = inip;
        delta = (8*rand + 2)/100 * norm(x0_init);
        [dimA, A, b] = datas(n, m, delta);
        for s = 1:skip
            for r = 1:100
                xd = l + (u - l) .* rand(n, 1); %#ok<NASGU>
            end
        end
        it8 = zeros(8, 1);
        for run = 1:8
            x0 = l + (u - l) .* rand(n, 1);
            try
                [~, ~, ~, iter, ~, ~, ~] = evalc('ProxGrad(n,m,l,u,x0,3,dimA,A,b)');
            catch ME
                iter = -1;
            end
            it8(run) = iter;
        end
        if isequal(it8, it_ref)
            offset = skip; break
        end
    end
    fprintf('offset=%d\n', offset);

    rng(123);
    [n, m, l, u, x0_init] = inip;
    delta = (8*rand + 2)/100 * norm(x0_init);
    [dimA, A, b] = datas(n, m, delta);
    for s = 1:offset
        for r = 1:100
            xd = l + (u - l) .* rand(n, 1); %#ok<NASGU>
        end
    end
    cnt = zeros(100, 1); it = zeros(100, 1); inf = zeros(100, 1);
    t0 = tic;
    for run = 1:100
        x0 = l + (u - l) .* rand(n, 1);
        QPSOLVECOUNT = 0;
        try
            [~, ~, info, iter, ~, ~, ~] = evalc('ProxGrad(n,m,l,u,x0,3,dimA,A,b)');
        catch ME
            info = -99; iter = -1;
        end
        cnt(run) = QPSOLVECOUNT; it(run) = iter; inf(run) = info;
        if mod(run, 20) == 0
            fprintf('  run %3d  elapsed %.1f s\n', run, toc(t0));
        end
    end
    ok = (inf == 0);
    if any(ok)
        fprintf('%-6s medQP=%6.1f medIter=%6.1f ratio=%5.3f match=%d\n', prob_name, ...
            median(cnt(ok)), median(it(ok)), median(cnt(ok)) / max(median(it(ok)), 1), ...
            isequal(it, R.prob_results{3}.iter_vals));
    else
        fprintf('%-6s no successful run\n', prob_name);
    end
    if ~exist('qp_out2', 'dir'), mkdir('qp_out2'); end
    out = struct('problem', prob_name, 'algorithm', 'MPG-Implicit', ...
        'qp_count', cnt, 'iter', it, 'info', inf, 'offset', offset);
    save(fullfile('qp_out2', [prob_name '_qp.mat']), 'out');
    fprintf('QP ONE DONE\n');
end
