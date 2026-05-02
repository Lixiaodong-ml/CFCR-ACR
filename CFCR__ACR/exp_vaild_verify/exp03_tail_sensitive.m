% Exp03: tail-sensitive distortion and low/high-tail subset analysis.
% Claim: tail mass mismatch is an independent source of fusion failure.
clc;
clear;

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();
cfg = mech.utils.defaultConfig();
cfg.seed = 303;
out_dir = mech.utils.makeOutputDir(cfg, 'exp03_tail_sensitive');

[U, Pi, y, truth] = mech.utils.makeSynthetic('tail', cfg);
rows = struct([]);
for rep = 1:cfg.num_repeats
    rng(cfg.seed + rep - 1);
    res = mech.utils.runCoreMethods(U, Pi, y, cfg);

    names = res.methods;
    Qs = res.Qs;
    rep_rows = [ ...
        mech.utils.subsetRows('low_tail', Qs, names, Pi, y, ~truth.high_tail, cfg.n_bins), ...
        mech.utils.subsetRows('high_tail', Qs, names, Pi, y, truth.high_tail, cfg.n_bins) ...
    ];
    for r = 1:numel(rep_rows)
        rep_rows(r).repeat = rep;
    end
    rows = [rows, rep_rows]; %#ok<AGROW>
end

summary_rows = mech.utils.writeRepeatedTables(rows, fullfile(out_dir, 'tail_subset_results.csv'), {'subset', 'method'});
disp(struct2table(summary_rows));

T = struct2table(rows);
T = T(T.repeat == 1, :);
figure('Color', 'w');
bar(categorical(T.method + "_" + T.subset), T.hellinger);
ylabel('Mean Hellinger error to latent consensus');
title('Tail-sensitive distortion: low-tail vs high-tail');
xtickangle(35);
saveas(gcf, fullfile(out_dir, 'tail_subset_hellinger.png'));
