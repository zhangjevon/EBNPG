% Added for the EBNPG manuscript.
% SPDX-License-Identifier: GPL-3.0-or-later
function r2_verify_k0()
% Verify that the R2 rerun of EBNPG-N2 on MGH33 (slot 8 of results/) reproduces
% the paired Euclidean run K0 of the fixed-kernel study (ne_full/MGH33_N2.mat),
% which was generated from the same stream and the same code.  If the two agree,
% the stored fixed-kernel block is canonical and is kept as is; otherwise the
% block must be regenerated with rerun_offset('MGH33', 7).
    R = load(fullfile('results', 'MGH33_results.mat'));
    it8 = R.prob_results{8}.iter_vals;
    i8  = R.prob_results{8}.info_vals;
    N = load(fullfile('ne_full', 'MGH33_N2.mat'));
    k0 = N.R.K0;
    same_info = isequal(i8, k0.info);
    same_iter = isequal(it8, k0.iter);
    fprintf('MGH33 rerun EBNPG-N2: SR=%.0f%%  medIt=%.1f\n', ...
        100*mean(i8==0), median(it8(i8==0)));
    fprintf('MGH33 fixed-kernel K0: SR=%.0f%%  medIt=%.1f\n', ...
        100*mean(k0.info==0), median(k0.iter(k0.info==0)));
    fprintf('info vectors identical: %d ; iter vectors identical: %d\n', ...
        same_info, same_iter);
    if same_info && same_iter
        fprintf('=> K0 block is canonical: ne_full/MGH33_N2.mat kept unchanged.\n');
    else
        fprintf('=> MISMATCH: regenerate with rerun_offset(''MGH33'', 7).\n');
    end
end
