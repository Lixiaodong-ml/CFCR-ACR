% Exp02: boundary-dependent distortion with easy/hard split.
% Claim: gains concentrate on boundary-uncertain hard instances.
clc;
clear;

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();
cfg = mech.utils.defaultConfig();
cfg.seed = 202;
out_dir = mech.utils.makeOutputDir(cfg, 'exp02_boundary_easy_hard');

[U, Pi, y, truth] = mech.utils.makeSynthetic('boundary', cfg);
rows = struct([]);
for rep = 1:cfg.num_repeats
    rng(cfg.seed + rep - 1);
    res = mech.utils.runCoreMethods(U, Pi, y, cfg);

    names = res.methods;
    Qs = res.Qs;
    rep_rows = [ ...
        mech.utils.subsetRows('easy_high_gap', Qs, names, Pi, y, ~truth.hard, cfg.n_bins), ...
        mech.utils.subsetRows('hard_low_gap', Qs, names, Pi, y, truth.hard, cfg.n_bins) ...
    ];
    for r = 1:numel(rep_rows)
        rep_rows(r).repeat = rep;
    end
    rows = [rows, rep_rows]; %#ok<AGROW>
end

summary_rows = mech.utils.writeRepeatedTables(rows, fullfile(out_dir, 'boundary_easy_hard_results.csv'), {'subset', 'method'});
disp(struct2table(summary_rows));

T = struct2table(rows);
T = T(T.repeat == 1, :);
figure('Color', 'w');
bar(categorical(T.method + "_" + T.subset), T.hellinger);
ylabel('Mean Hellinger error to latent consensus');
title('Boundary distortion: easy vs hard');
xtickangle(35);
saveas(gcf, fullfile(out_dir, 'boundary_easy_hard_hellinger.png'));
