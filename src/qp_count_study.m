% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function qp_count_study(phase)
% Count the number of proximal (QP) subproblem solves.
% phase = 'verify' : small subsample, checks the relation between the number
%                    of QP solves and the iteration counter for each family
% phase = 'implicit' : full 42x100 re-run of MPG-Implicit only
    addpath(fileparts(mfilename('fullpath')));
    if ~exist('results','dir'), mkdir('results'); end
    global problem QPSOLVECOUNT

    probs_all = {'AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1', ...
        'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2', ...
        'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1', ...
        'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1'};

    modes = {'N0','N1','N2','N3','N5','N6','N7'};
    % slot order exactly as in run_problem.m
    slots = {};
    slots{1} = struct('kind','mpg','lsopt',1,'name','MPG-Explicit');
    slots{2} = struct('kind','mpg','lsopt',2,'name','MPG-Armijo');
    slots{3} = struct('kind','mpg','lsopt',3,'name','MPG-Implicit');
    slots{4} = struct('kind','cnpg','mode','max','name','CNPG-Max');
    slots{5} = struct('kind','cnpg','mode','avg','name','CNPG-Avg');
    for mi = 1:7
        slots{5+mi} = struct('kind','ebnpg','mode',modes{mi},'name',['EBNPG-' modes{mi}]);
    end

    if strcmp(phase, 'verify')
        probs = {'AP1','BK1','DTLZ2','MGH16','MGH33','NIC1','NSP1','ZDT6'};
        target = [1 2 3 4 5 8];   % MPG x3, CNPG x2, EBNPG-N2
        nruns = 10;
    else
        probs = probs_all;
        target = 3;               % MPG-Implicit only
        nruns = 100;
    end

    fprintf('phase=%s  nprobs=%d  nruns=%d  slots=%s\n', phase, length(probs), nruns, mat2str(target));

    for pi = 1:length(probs)
        prob_name = probs{pi};
        problem = prob_name;
        rng(123);
        [n, m, l, u, x0_init] = inip;
        delta = (8*rand + 2)/100 * norm(x0_init);
        [dimA, A, b] = datas(n, m, delta);

        for ai = 1:length(slots)
            % replay the starting points of this slot exactly as run_problem.m does
            X0 = zeros(n, nruns);
            for run = 1:nruns
                X0(:, run) = l + (u - l) .* rand(n, 1);
            end
            if ~ismember(ai, target)
                % still consume the remaining draws so later slots stay aligned
                for run = nruns+1:100
                    xdummy = l + (u - l) .* rand(n, 1); %#ok<NASGU>
                end
                continue
            end
            s = slots{ai};
            cnt = zeros(nruns,1); it = zeros(nruns,1); inf = zeros(nruns,1);
            for run = 1:nruns
                x0 = X0(:, run);
                QPSOLVECOUNT = 0;
                try
                    if strcmp(s.kind,'mpg')
                        [~, ~, info, iter, ~, ~, ~] = evalc('ProxGrad(n,m,l,u,x0,s.lsopt,dimA,A,b)');
                    elseif strcmp(s.kind,'cnpg')
                        [~, info, iter, ~, ~, ~] = CNPG(n, m, l, u, x0, 1e-4, 0.5, 1e-4, 200, s.mode, A, b, dimA, 'M', 5, 'theta_avg', 0.5);
                    else
                        [~, info, iter, ~, ~, ~] = run_ebnpg_silent(n,m,l,u,x0,0.5,1e-4,0.5,1.0,1e-4,200,eye(n),A,b,dimA,s.mode,5,0.9,0.5);
                    end
                catch ME
                    info = -99; iter = -1;
                end
                cnt(run) = QPSOLVECOUNT; it(run) = iter; inf(run) = info;
            end
            ok = (inf == 0);
            if any(ok)
                fprintf('%-6s %-13s  median #QP=%6.1f  median iter=%6.1f  ratio=%5.3f  (cnt-iter: %s)\n', ...
                    prob_name, s.name, median(cnt(ok)), median(it(ok)), median(cnt(ok))/max(median(it(ok)),1), ...
                    mat2str(unique(cnt(ok)-it(ok))'));
            else
                fprintf('%-6s %-13s  no successful run\n', prob_name, s.name);
            end
            if strcmp(phase,'implicit')
                out = struct('problem',prob_name,'algorithm',s.name,'qp_count',cnt,'iter',it,'info',inf);
                if ~exist('qp_out2','dir'), mkdir('qp_out2'); end
                save(fullfile('qp_out2',[prob_name '_qp.mat']),'out');
            end
        end
    end
    fprintf('QP STUDY DONE (%s)\n', phase);
end
