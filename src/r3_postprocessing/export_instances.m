% Added for the EBNPG manuscript (third revision).
% SPDX-License-Identifier: GPL-3.0-or-later
function export_instances(prob_list, out_dir)
% EXPORT_INSTANCES  Write the instance data that the experiments consume, so
% that every number in the paper can be checked without regenerating the
% random stream.
%
% Produces, in <out_dir> (default ../archive):
%   instances.csv        prob, n, m, delta, l, u, x_ref (';' separated)
%   kernel_samples.csv   prob, k, j, x_j        (the 20 fixed-kernel samples)
%   kernel_weights.csv   prob, j, w_j           (the resulting weights)
%   B_matrices.mat       struct B with B{ip,i} = B_i for objective i
%
% x_ref is the point returned by inip.m after rng(123) (Subsection 6.2), and
% delta is the draw that follows it, exactly as in run_problem.m.  The kernel
% samples are drawn with rng(7) inside curvature_weights (noneuclid_run.m),
% so they are independent of the run stream.

if nargin < 2 || isempty(out_dir)
    out_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'archive');
end
if ~exist(out_dir,'dir'), mkdir(out_dir); end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'Code'));
global problem

fid = fopen(fullfile(out_dir,'instances.csv'),'w');
fprintf(fid,'prob,n,m,delta,l,u,x_ref\n');
fk  = fopen(fullfile(out_dir,'kernel_samples.csv'),'w');
fprintf(fk,'prob,k,j,x\n');
fw  = fopen(fullfile(out_dir,'kernel_weights.csv'),'w');
fprintf(fw,'prob,j,w\n');
B = struct('prob',{},'B',{});

for pi = 1:numel(prob_list)
    p = prob_list{pi};
    rng(123); problem = p;
    [n,m,l,u,xref] = inip;
    delta = (8*rand + 2)/100 * norm(xref);
    [dimA,A,b] = datas(n,m,delta);

    fprintf(fid,'%s,%d,%d,%.12g,%s,%s,%s\n', p, n, m, delta, ...
            vec2str(l), vec2str(u), vec2str(xref));

    K = 20; rng(7); P = l + (u-l).*rand(n,K);
    for k = 1:K
        for j = 1:n
            fprintf(fk,'%s,%d,%d,%.12g\n', p, k, j, P(j,k));
        end
    end
    w = curvature_weights_from(n,m,l,u,P);
    for j = 1:n, fprintf(fw,'%s,%d,%.10g\n', p, j, w(j)); end

    B(end+1).prob = p;                                      %#ok<AGROW>
    Bi = cell(m,1);
    for i = 1:m, Bi{i} = A{i}(1:n,:); end
    B(end).B = Bi;
end
fclose(fid); fclose(fk); fclose(fw);
save(fullfile(out_dir,'B_matrices.mat'),'B');
fprintf('instances: %d problems written to %s\n', numel(prob_list), out_dir);
end

% ------------------------------------------------------------------------
function s = vec2str(v)
s = strjoin(arrayfun(@(z) sprintf('%.12g',z), v(:)', 'UniformOutput', false), ';');
end

% ------------------------------------------------------------------------
function w = curvature_weights_from(n,m,l,u,P)
% Same arithmetic as curvature_weights in noneuclid_run.m, but on a given
% sample set, so that the archived samples and weights are consistent.
K = size(P,2);
d = zeros(n,1); hscale = 1e-3;
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
            f0 = evalg(n, xk, i); fp = evalg(n, xp, i); fm = evalg(n, xm, i);
            hj2 = xp(j) - xm(j);
            acc = acc + abs((fp - 2*f0 + fm) / (hj2*hj2/4));
        end
        d(j) = d(j) + acc / m;
    end
end
d = d / K; d = max(d, 1e-10);
w = d / mean(d); w = min(max(w, 1e-2), 1e2);
end
