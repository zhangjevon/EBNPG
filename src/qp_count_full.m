% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function qp_count_full()
% Count the proximal (QP) solves of MPG-Implicit for all 42 problems.
% The random-stream offset of the stored run is detected automatically by
% matching the first 8 iteration counts, so the QP counts refer to exactly the
% same 100 starting points as the stored results.
    addpath(fileparts(mfilename('fullpath')));
    if ~exist('results','dir'), mkdir('results'); end
    global problem QPSOLVECOUNT

    probs = {'AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1', ...
        'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2', ...
        'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1', ...
        'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1'};

    if ~exist('qp_out2','dir'), mkdir('qp_out2'); end

    for pi = 1:length(probs)
        prob_name = probs{pi};
        outfile = fullfile('qp_out2', [prob_name '_qp.mat']);
        if exist(outfile, 'file')
            fprintf('SKIP %-6s (already done)\n', prob_name);
            continue
        end
        problem = prob_name;
        R = load(fullfile('results', [prob_name '_results.mat']));
        it_ref = R.prob_results{3}.iter_vals(1:8);

        % ---- detect the stream offset -------------------------------------
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
            fprintf('%-6s  OFFSET NOT FOUND\n', prob_name);
            continue
        end

        % ---- full 100-run pass with the detected offset -------------------
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
        if any(ok)
            fprintf('%-6s offset=%d  medQP=%6.1f  medIter=%6.1f  ratio=%5.3f  match=%d\n', ...
                prob_name, offset, median(cnt(ok)), median(it(ok)), ...
                median(cnt(ok)) / max(median(it(ok)), 1), isequal(it, R.prob_results{3}.iter_vals));
        else
            fprintf('%-6s offset=%d  no successful run\n', prob_name, offset);
        end
        out = struct('problem', prob_name, 'algorithm', 'MPG-Implicit', ...
            'qp_count', cnt, 'iter', it, 'info', inf, 'offset', offset);
        save(outfile, 'out');
    end
    fprintf('QP COUNT FULL DONE\n');
end
