% Exp05: disagreement-conditioned gain.
% Claim: recovery helps most when raw oracle posteriors disagree.
clc;
clear;

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();
cfg = mech.utils.defaultConfig();
cfg.seed = 505;
out_dir = mech.utils.makeOutputDir(cfg, 'exp05_disagreement_conditioned_gain');

[U, Pi, y] = mech.utils.makeSynthetic('mixed', cfg);
[n, M, ~] = size(U);
dis = zeros(n, 1);
cnt = 0;
for a = 1:M
    for b = (a + 1):M
        dis = dis + mech.utils.hellingerRows(squeeze(U(:, a, :)), squeeze(U(:, b, :)));
        cnt = cnt + 1;
    end
end
dis = dis / cnt;
edges = quantile(dis, [0, 1/3, 2/3, 1]);
labels = {'low_disagreement', 'mid_disagreement', 'high_disagreement'};

rows = struct([]);
for rep = 1:cfg.num_repeats
    rng(cfg.seed + rep - 1);
    res = mech.utils.runCoreMethods(U, Pi, y, cfg);

    for g = 1:3
        if g < 3
            mask = dis >= edges(g) & dis < edges(g + 1);
        else
            mask = dis >= edges(g) & dis <= edges(g + 1);
        end
        rep_rows = mech.utils.subsetRows(labels{g}, res.Qs, res.methods, Pi, y, mask, cfg.n_bins);
        for r = 1:numel(rep_rows)
            rep_rows(r).repeat = rep;
        end
        rows = [rows, rep_rows]; %#ok<AGROW>
    end
end

summary_rows = mech.utils.writeRepeatedTables(rows, fullfile(out_dir, 'disagreement_conditioned_results.csv'), {'subset', 'method'});
disp(struct2table(summary_rows));

T = struct2table(rows);
T = T(T.repeat == 1, :);
figure('Color', 'w');
bar(categorical(T.method + "_" + T.subset), T.hellinger);
ylabel('Mean Hellinger error to latent consensus');
title('Recovery error by raw inter-oracle disagreement');
xtickangle(35);
saveas(gcf, fullfile(out_dir, 'disagreement_conditioned_hellinger.png'));
