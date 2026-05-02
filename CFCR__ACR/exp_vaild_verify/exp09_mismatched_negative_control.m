% Exp09: mismatched negative controls.
% Claim: gains should degrade outside the scalar radial distortion family.

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();
cfg = mech.utils.defaultConfig();
cfg.seed = 909;
out_dir = mech.utils.makeOutputDir(cfg, 'exp09_mismatched_negative_control');

scenarios = {'global', 'anisotropic', 'nonmonotone'};
mismatch_values = [0.0, 0.75, 0.75];
all_rows = struct([]);

for s = 1:numel(scenarios)
    cfg_s = cfg;
    cfg_s.mismatch_strength = mismatch_values(s);
    [U, Pi, y, truth] = mech.utils.makeSynthetic(scenarios{s}, cfg_s);
    for rep = 1:cfg.num_repeats
        rng(cfg_s.seed + rep - 1);
        res = mech.utils.runCoreMethods(U, Pi, y, cfg_s);
        rows = res.rows;
        for r = 1:numel(rows)
            rows(r).scenario = string(truth.scenario);
            rows(r).mismatch_strength = cfg_s.mismatch_strength;
            rows(r).repeat = rep;
        end
        all_rows = [all_rows, rows]; %#ok<AGROW>
    end
end

summary_rows = mech.utils.writeRepeatedTables(all_rows, fullfile(out_dir, 'mismatched_negative_control_results.csv'), {'scenario', 'mismatch_strength', 'method'});
disp(struct2table(summary_rows));
