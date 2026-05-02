% Exp10: approximate mismatch strength scan.
% Claim: recovery degrades gradually under approximate mismatch.
clc;
clear;

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();
cfg = mech.utils.defaultConfig();
cfg.seed = 1001;
cfg.n = 600;
out_dir = mech.utils.makeOutputDir(cfg, 'exp10_approximate_mismatch_scan');

strengths = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90];
all_rows = struct([]);

for j = 1:numel(strengths)
    cfg_j = cfg;
    cfg_j.mismatch_strength = strengths(j);
    [U, Pi, y] = mech.utils.makeSynthetic('approx_mismatch', cfg_j);
    for rep = 1:cfg.num_repeats
        rng(cfg_j.seed + rep - 1);
        res = mech.utils.runCoreMethods(U, Pi, y, cfg_j);
        rows = res.rows;
        for r = 1:numel(rows)
            rows(r).mismatch_strength = strengths(j);
            rows(r).repeat = rep;
        end
        all_rows = [all_rows, rows]; %#ok<AGROW>
    end
end

summary_rows = mech.utils.writeRepeatedTables(all_rows, fullfile(out_dir, 'approximate_mismatch_scan_results.csv'), {'mismatch_strength', 'method'});
disp(struct2table(summary_rows));

T = struct2table(all_rows);
T = T(T.repeat == 1, :);
figure('Color', 'w');
hold on;
methods = unique(T.method, 'stable');
for m = 1:numel(methods)
    mask = T.method == methods(m);
    plot(T.mismatch_strength(mask), T.hellinger(mask), '-o', 'DisplayName', methods(m));
end
xlabel('Mismatch strength');
ylabel('Mean Hellinger error to latent consensus');
legend('Location', 'best');
title('Approximate mismatch scan');
saveas(gcf, fullfile(out_dir, 'approximate_mismatch_scan.png'));
