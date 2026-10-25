% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function r2_qp_mgh33()
% Re-measure the proximal (QP) solve counts of MPG-Implicit on MGH33 after the
% R2 rerun.  Mirrors qp_count_full.m for the single problem MGH33: the stream
% offset is detected by matching the first eight stored iteration counts.
    global problem QPSOLVECOUNT
    prob_name = 'MGH33';
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
            offset = skip;
            break
        end
    end
    if offset < 0
        error('offset not found for %s', prob_name);
    end

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
    for run = 1:100
        x0 = l + (u - l) .* rand(n, 1);
        QPSOLVECOUNT = 0;
        try
            [~, ~, info, iter, ~, ~, ~] = evalc('ProxGrad(n,m,l,u,x0,3,dimA,A,b)');
        catch ME
            info = -99; iter = -1;
        end
        cnt(run) = QPSOLVECOUNT; it(run) = iter; inf(run) = info;
    end
    ok = (inf == 0);
    out = struct('problem', prob_name, 'algorithm', 'MPG-Implicit', ...
        'qp_count', cnt, 'iter', it, 'info', inf, 'offset', offset);
    save(fullfile('qp_out2', [prob_name '_qp.mat']), 'out');
    fprintf('%s offset=%d  medQP=%.1f  medIter=%.1f  ratio=%.3f  match=%d\n', ...
        prob_name, offset, median(cnt(ok)), median(it(ok)), ...
        median(cnt(ok)) / max(median(it(ok)), 1), isequal(it, R.prob_results{3}.iter_vals));
end
