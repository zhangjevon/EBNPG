% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
% Export the reproducible part of the numerical experiments.
%
% For each of the 42 test problems this writes archive/<problem>.mat containing
%   n, m, l, u        problem dimensions and box
%   delta             threshold of the polyhedral support functions
%   B{1..m}           the matrices defining g_i  (A_i = [B_i; -B_i])
%   starts{1..12}     the 100 starting points of each algorithm slot (n x 100)
%
% The generation order reproduces run_problem.m exactly:
%   rng(123); inip; delta = (8*rand+2)/100*norm(x0_init); datas; then twelve
%   blocks of 100 draws of rand(n,1), one block per algorithm slot, with the
%   block map of run_problem.m (natural protocol, except for the four
%   interrupted sessions AP4, BK1, PNR and DGO2).
function export_archive()
    probs = {'AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS', ...
             'Hil1','IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26', ...
             'MGH33','MLF2','MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1', ...
             'NIC2','NIC3','NIM1','NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4', ...
             'Toi9','VU2','ZDT6','ZLT1'};
    nalg = 12;
    nruns = 100;
    outdir = 'archive';
    if ~exist(outdir,'dir'), mkdir(outdir); end

    fid = fopen(fullfile(outdir,'instances.csv'),'w');
    fprintf(fid,'problem,n,m,delta,l_min,u_max\n');

    for pi = 1:numel(probs)
        pname = probs{pi};
        clear global problem
        global problem
        problem = pname;

        rng(123);
        [n,m,l,u,x0_init] = inip;              %#ok<ASGLU>
        delta = (8*rand+2)/100 * norm(x0_init);
        [dimA,A,b] = datas(n,m,delta);         %#ok<ASGLU>

        B = cell(m,1);
        for i = 1:m
            B{i} = A{i}(1:n,:);                % A_i = [B_i; -B_i]
        end

        % Block map, identical to run_problem.m: slot a consumes block a under
        % the natural protocol; the four problems whose original sessions were
        % interrupted and resumed without advancing the stream deviate from it
        % (AP4, BK1, PNR after algorithm 1, DGO2 after algorithm 3), and the
        % map reproduces the stored assignment of every problem exactly.
        blk = 1:nalg;
        if ismember(pname, {'AP4','BK1','PNR'})
            blk = [1 1 2:(nalg-1)];
        elseif ismember(pname, {'DGO2'})
            blk = [1 2 3 1:9];
        end
        blocks = cell(nalg, 1);
        for a = 1:nalg
            S = zeros(n, nruns);
            for r = 1:nruns
                S(:,r) = l + (u-l).*rand(n,1);
            end
            blocks{a} = S;
        end
        starts = cell(nalg, 1);
        for a = 1:nalg
            starts{a} = blocks{blk(a)};
        end

        save(fullfile(outdir,[pname '.mat']), ...
             'n','m','l','u','delta','B','starts','-v7.3');
        fprintf(fid,'%s,%d,%d,%.17g,%.17g,%.17g\n', pname, n, m, delta, ...
                min(l), max(u));
        fprintf('%-6s n=%3d m=%d delta=%.6g  exported\n', pname, n, m, delta);
    end
    fclose(fid);
    fprintf('\nDone: %d problems written to %s\n', numel(probs), outdir);
end
