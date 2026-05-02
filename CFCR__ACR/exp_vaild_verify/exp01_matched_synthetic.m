% Exp01: matched synthetic main experiment.
% Claim: semantic alignment alone does not guarantee geometric comparability.
clc;
clear;

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();
cfg = mech.utils.defaultConfig();
cfg.seed = 101;
out_dir = mech.utils.makeOutputDir(cfg, 'exp01_matched_synthetic');

scenarios = {'global', 'boundary', 'tail', 'mixed'};
all_rows = struct([]);

for s = 1:numel(scenarios)
    [U, Pi, y, truth] = mech.utils.makeSynthetic(scenarios{s}, cfg);
    for rep = 1:cfg.num_repeats
        rng(cfg.seed + rep - 1);
        res = mech.utils.runCoreMethods(U, Pi, y, cfg);
        rows = res.rows;
        for r = 1:numel(rows)
            rows(r).scenario = string(truth.scenario);
            rows(r).repeat = rep;
        end
        all_rows = [all_rows, rows]; %#ok<AGROW>
    end
end

summary_rows = mech.utils.writeRepeatedTables(all_rows, fullfile(out_dir, 'matched_synthetic_results.csv'), {'scenario', 'method'});
disp(struct2table(summary_rows));
