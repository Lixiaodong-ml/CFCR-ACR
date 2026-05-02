% Exp07: scalar versus coordinate-wise rectification.
% Claim: scalar rectification is semantically conservative and order-preserving.

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();
cfg = mech.utils.defaultConfig();
cfg.seed = 707;
cfg.max_iterations = 15;
out_dir = mech.utils.makeOutputDir(cfg, 'exp07_scalar_vs_coordinatewise');

[U, Pi, y] = mech.utils.makeSynthetic('mixed', cfg);
rows = struct([]);
for rep = 1:cfg.num_repeats
    rng(cfg.seed + rep - 1);

    [Q_scalar, ~, R_scalar] = mech.utils.customRecovery(U, 'frozen', [1 1 1 1], cfg, 'hellinger');
    [Q_coord, R_coord] = mech.utils.coordinateWiseRecovery(U, cfg);

    row_scalar = mech.utils.scoreConsensus('Scalar', Q_scalar, Pi, y, cfg.n_bins);
    row_scalar.order_reversal_rate = mech.utils.orderReversalRate(U, R_scalar);
    row_scalar.repeat = rep;
    row_coord = mech.utils.scoreConsensus('CoordinateWise', Q_coord, Pi, y, cfg.n_bins);
    row_coord.order_reversal_rate = mech.utils.orderReversalRate(U, R_coord);
    row_coord.repeat = rep;
    rows = [rows, row_scalar, orderfields(row_coord, row_scalar)]; %#ok<AGROW>
end

summary_rows = mech.utils.writeRepeatedTables(rows, fullfile(out_dir, 'scalar_vs_coordinatewise_results.csv'), {'method'});
disp(struct2table(summary_rows));
