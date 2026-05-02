% Exp04: before/after geometry diagnostics.
% Claim: comparability recovery reduces observable geometric disagreement.
clc;
clear;

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();
cfg = mech.utils.defaultConfig();
cfg.seed = 404;
out_dir = mech.utils.makeOutputDir(cfg, 'exp04_geometry_diagnostics');

[U, Pi, y] = mech.utils.makeSynthetic('mixed', cfg); %#ok<ASGLU>
rows = struct([]);
for rep = 1:cfg.num_repeats
    rng(cfg.seed + rep - 1);
    res = mech.utils.runCoreMethods(U, Pi, y, cfg);

    raw_stats = mech.utils.geometryStats(U);
    raw_stats.stage = string('raw');
    raw_stats.repeat = rep;
    rep_rows = raw_stats;

    k = 2;
    if ~isempty(res.R_cfcr)
        cfcr_stats = mech.utils.geometryStats(res.R_cfcr);
        cfcr_stats.stage = string('after_CFCR');
        cfcr_stats.repeat = rep;
        rep_rows(k) = orderfields(cfcr_stats, rep_rows);
        k = k + 1;
    end
    if ~isempty(res.R_acr)
        acr_stats = mech.utils.geometryStats(res.R_acr);
        acr_stats.stage = string('after_ACR');
        acr_stats.repeat = rep;
        rep_rows(k) = orderfields(acr_stats, rep_rows);
    end
    rows = [rows, rep_rows]; %#ok<AGROW>
end

summary_rows = mech.utils.writeRepeatedTables(rows, fullfile(out_dir, 'geometry_diagnostics.csv'), {'stage'});
disp(struct2table(summary_rows));

T = struct2table(rows);
T = T(T.repeat == 1, :);
figure('Color', 'w');
tiledlayout(2, 2);
nexttile; bar(categorical(T.stage), T.entropy_oracle_std); title('Entropy oracle std');
nexttile; bar(categorical(T.stage), T.gap_oracle_std); title('Top-2 gap oracle std');
nexttile; bar(categorical(T.stage), T.tail_oracle_std); title('Tail oracle std');
nexttile; bar(categorical(T.stage), T.pairwise_hellinger); title('Pairwise Hellinger');
saveas(gcf, fullfile(out_dir, 'geometry_diagnostics.png'));
