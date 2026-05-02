% Exp08: Hellinger barycenter versus Euclidean and KL-style alternatives.
% Claim: the barycenter geometry matters after rectification.

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();
cfg = mech.utils.defaultConfig();
cfg.seed = 808;
cfg.max_iterations = 18;
out_dir = mech.utils.makeOutputDir(cfg, 'exp08_hellinger_vs_alternatives');

objectives = {'hellinger', 'euclidean', 'kl'};
[U, Pi, y] = mech.utils.makeSynthetic('mixed', cfg);
rows = struct([]);
for rep = 1:cfg.num_repeats
    rng(cfg.seed + rep - 1);
    rep_rows = struct([]);
    for j = 1:numel(objectives)
        [Q, ~, ~] = mech.utils.customRecovery(U, 'frozen', [1 1 1 1], cfg, objectives{j});
        row_j = mech.utils.scoreConsensus(['Rectifier_' objectives{j}], Q, Pi, y, cfg.n_bins);
        row_j.repeat = rep;
        if j == 1
            rep_rows = row_j;
        else
            rep_rows(j) = orderfields(row_j, rep_rows);
        end
    end
    rows = [rows, rep_rows]; %#ok<AGROW>
end

summary_rows = mech.utils.writeRepeatedTables(rows, fullfile(out_dir, 'barycenter_alternatives_results.csv'), {'method'});
disp(struct2table(summary_rows));
