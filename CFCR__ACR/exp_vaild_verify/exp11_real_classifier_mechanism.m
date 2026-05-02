% Exp11: real black-box classifier ensemble mechanism analysis.
% Claim: the same diagnostics can be run on real aligned soft outputs.
clc;
clear;

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();
cfg = mech.utils.defaultConfig();
base_seed = cfg.seed;
out_dir = mech.utils.makeOutputDir(cfg, 'exp11_real_classifier_mechanism');

script_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(script_dir);
input_dir = fullfile(root_dir, 'RCMO_datasets', 'RCMO_input');
files = dir(fullfile(input_dir, '*.mat'));
if isempty(files)
    warning('No real .mat datasets found under %s. Skipping exp11.', input_dir);
    return;
end

all_rows = struct([]);
geom_rows = struct([]);
consensus_geom_rows = struct([]);

for rep = 1:cfg.num_repeats
    cfg_rep = cfg;
    cfg_rep.seed = base_seed + rep - 1;
    rng(cfg_rep.seed);

    for f = 1:numel(files)
        path = fullfile(files(f).folder, files(f).name);
        S = load(path);
        [U, y, ok] = localLoadSoftDataset(S);
        if ~ok
            warning('Skipping %s because no compatible U/y variables were found.', files(f).name);
            continue;
        end
        U = max(U, cfg_rep.epsilon);
        U = U ./ sum(U, 3);

        res = mech.utils.runCoreMethods(U, [], y, cfg_rep);
        rows = res.rows;
        for r = 1:numel(rows)
            rows(r).dataset = string(files(f).name);
            rows(r).repeat = rep;
        end
        if isempty(all_rows)
            all_rows = rows;
        else
            all_rows = [all_rows, orderfields(rows, all_rows)]; %#ok<AGROW>
        end

        % Oracle-level geometry diagnostics. Mean / GeoMean / HellingerRaw do
        % not rectify per-oracle outputs, so their oracle geometry is identical
        % to raw. CFCR and ACR are the methods that actually produce rectified
        % per-oracle outputs.
        raw_stats = localOracleGeometryRow(U, files(f).name, 'Raw', ...
            'raw_original', false);
        raw_stats.repeat = rep;
        geom_rows = localAppendRows(geom_rows, raw_stats);

        for r = 1:numel(res.methods)
            method_name = res.methods{r};
            switch method_name
                case 'CFCR'
                    if isempty(res.R_cfcr)
                        continue;
                    end
                    oracle_outputs_for_method = res.R_cfcr;
                    stage_name = 'after_CFCR';
                    has_rectification = true;
                case 'ACR'
                    if isempty(res.R_acr)
                        continue;
                    end
                    oracle_outputs_for_method = res.R_acr;
                    stage_name = 'after_ACR';
                    has_rectification = true;
                otherwise
                    oracle_outputs_for_method = U;
                    stage_name = 'no_rectification_raw_oracles';
                    has_rectification = false;
            end
            method_stats = localOracleGeometryRow(oracle_outputs_for_method, ...
                files(f).name, method_name, stage_name, has_rectification);
            method_stats.repeat = rep;
            geom_rows = localAppendRows(geom_rows, method_stats);
        end

        % Consensus-level geometry for every fusion method. This is different
        % from oracle-level comparability: it describes the final fused posterior.
        for r = 1:numel(res.methods)
            consensus_stats = localConsensusGeometryRow(res.Qs{r}, ...
                files(f).name, res.methods{r});
            consensus_stats.repeat = rep;
            consensus_geom_rows = localAppendRows(consensus_geom_rows, consensus_stats);
        end
    end
end

if ~isempty(all_rows)
    mech.utils.writeRepeatedTables(all_rows, fullfile(out_dir, 'real_classifier_results.csv'), {'dataset', 'method'});
    disp(struct2table(all_rows));
end
if ~isempty(geom_rows)
    mech.utils.writeRepeatedTables(geom_rows, fullfile(out_dir, 'real_oracle_geometry_diagnostics.csv'), {'dataset', 'method', 'stage', 'has_rectification'});
    disp(struct2table(geom_rows));
end
if ~isempty(consensus_geom_rows)
    mech.utils.writeRepeatedTables(consensus_geom_rows, fullfile(out_dir, 'real_consensus_geometry.csv'), {'dataset', 'method'});
end

function [U, y, ok] = localLoadSoftDataset(S)
    ok = false;
    U = [];
    y = [];

    if isfield(S, 'P_cell') && iscell(S.P_cell) && isfield(S, 'gt')
        y = double(S.gt(:));
        if min(y) == 0
            y = y + 1;
        end
        n = numel(y);
        M = numel(S.P_cell);
        first_prob = localStandardizeProb(S.P_cell{1}, n, []);
        K = size(first_prob, 2);
        U = zeros(n, M, K);
        U(:, 1, :) = reshape(first_prob, n, 1, K);
        for m = 2:M
            prob_m = localStandardizeProb(S.P_cell{m}, n, K);
            U(:, m, :) = reshape(prob_m, n, 1, K);
        end
        ok = true;
        return;
    end

    y_names = {'y_true', 'Y', 'y', 'labels', 'label', 'gt'};
    for i = 1:numel(y_names)
        if isfield(S, y_names{i})
            y = S.(y_names{i});
            break;
        end
    end

    U_names = {'U', 'outputs', 'scores', 'probs', 'probabilities', 'all_probs'};
    for i = 1:numel(U_names)
        if isfield(S, U_names{i})
            U = S.(U_names{i});
            break;
        end
    end

    if isempty(U)
        fields = fieldnames(S);
        for i = 1:numel(fields)
            X = S.(fields{i});
            if isnumeric(X) && ndims(X) == 3
                U = X;
                break;
            end
        end
    end

    if isempty(y) || isempty(U)
        return;
    end

    y = double(y(:));
    U = double(U);
    dims = size(U);
    if numel(dims) ~= 3
        return;
    end

    % Convert common M x n x K layout to n x M x K if needed.
    if dims(1) ~= numel(y) && dims(2) == numel(y)
        U = permute(U, [2 1 3]);
    end
    if size(U, 1) ~= numel(y)
        return;
    end

    if min(y) == 0
        y = y + 1;
    end
    ok = true;
end

function prob = localStandardizeProb(prob, n_expected, K_expected)
    prob = double(prob);
    if size(prob, 1) ~= n_expected && size(prob, 2) == n_expected
        prob = prob';
    end
    if size(prob, 1) ~= n_expected
        error('Probability matrix row count does not match labels.');
    end
    if ~isempty(K_expected) && size(prob, 2) ~= K_expected
        error('Probability matrix class count mismatch across oracles.');
    end
    prob = max(prob, 1e-12);
    prob = prob ./ sum(prob, 2);
end

function row = localOracleGeometryRow(U, dataset_name, method_name, stage_name, has_rectification)
    row = mech.utils.geometryStats(U);
    row.dataset = string(dataset_name);
    row.method = string(method_name);
    row.stage = string(stage_name);
    row.has_rectification = logical(has_rectification);
end

function row = localConsensusGeometryRow(Q, dataset_name, method_name)
    K = size(Q, 2);
    Q_safe = max(Q, 1e-12);
    Q_safe = Q_safe ./ sum(Q_safe, 2);
    sorted_Q = sort(Q_safe, 2, 'descend');

    ent = -sum(Q_safe .* log(Q_safe), 2) / log(K);
    gap = sorted_Q(:, 1) - sorted_Q(:, 2);
    tail = 1 - sorted_Q(:, 1) - sorted_Q(:, 2);

    row = struct();
    row.dataset = string(dataset_name);
    row.method = string(method_name);
    row.consensus_entropy_mean = mean(ent);
    row.consensus_gap_mean = mean(gap);
    row.consensus_tail_mean = mean(tail);
    row.consensus_confidence_mean = mean(sorted_Q(:, 1));
end

function rows = localAppendRows(rows, new_rows)
    if isempty(rows)
        rows = new_rows;
    else
        rows = [rows, orderfields(new_rows, rows)]; %#ok<AGROW>
    end
end
