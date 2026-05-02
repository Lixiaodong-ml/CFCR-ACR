%% Entropy-aligned input preprocessing + B-continuation for CFCR / ACR
% This runner keeps the CFCR / ACR objective unchanged. It only changes
% the input preprocessing and the outer optimization path:
%   1) optional oracle-level entropy alignment before CFCR / ACR,
%   2) optional mild adaptive smoothing after entropy alignment,
%   3) continuation over an increasing B schedule.
%
% The purpose is to test stronger paper-compatible implementation choices
% without changing the theoretical objective or the scalar rectifier.

clearvars;
close all;
clc;

% =========================================================================
% SETTINGS
% =========================================================================

input_dir = fullfile('RCMO_datasets', 'RCMO_input');
dataset_files = dir(fullfile(input_dir, '*.mat'));

if isempty(dataset_files)
    error('No .mat files found under %s', input_dir);
end

% Choose one of: 'balanced', 'acc', 'nll', 'brier', 'ece', 'objective'.
selection_metric = 'balanced';

% Use inf for full search. Set small values for a quick smoke run.
max_datasets = inf;
max_configs = inf;
show_dataset_progress = true;

% Each B stage runs this many outer iterations.
stage_max_iterations = 25;
optimizer_max_iterations = 120;
tol = 1e-6;
n_bins = 10;
epsilon = 1e-8;

method_list = {'CFCR', 'ACR'};
preprocess_mode_list = {'none', 'entropy_align', 'entropy_align_smooth'};
theta_init_list = {'zero', 'entropy_match'};
acr_start_list = {'raw', 'cfcr'};

B_schedule_list = { ...
    [0.003 0.01 0.03], ...
    [0.003 0.01 0.03 0.1], ...
    [0.01 0.03 0.1 0.2] ...
};

entropy_gamma_grid = exp(linspace(log(0.2), log(5.0), 81));
smooth_strength = 0.01;

dataset_files = dataset_files(1:min(numel(dataset_files), max_datasets));
datasets = preload_datasets(dataset_files, n_bins);
baseline_summary = summarize_baselines(datasets);
config_grid = build_config_grid(method_list, preprocess_mode_list, ...
    theta_init_list, acr_start_list, B_schedule_list);
config_grid = config_grid(1:min(numel(config_grid), max_configs));

fprintf('\n============================================================\n');
fprintf('Entropy-aligned + B-continuation CFCR / ACR\n');
fprintf('Datasets      : %d\n', numel(datasets));
fprintf('Configurations: %d\n', numel(config_grid));
fprintf('Selection     : %s\n', selection_metric);
fprintf('============================================================\n\n');

fprintf('Arithmetic mean baseline: ACC %.4f | NLL %.4f | Brier %.4f | ECE %.4f\n', ...
    baseline_summary.mean.acc_mu, baseline_summary.mean.nll_mu, ...
    baseline_summary.mean.brier_mu, baseline_summary.mean.ece_mu);
fprintf('Geometric mean baseline : ACC %.4f | NLL %.4f | Brier %.4f | ECE %.4f\n\n', ...
    baseline_summary.geo.acc_mu, baseline_summary.geo.nll_mu, ...
    baseline_summary.geo.brier_mu, baseline_summary.geo.ece_mu);

results = struct([]);

for config_idx = 1:numel(config_grid)
    config = config_grid(config_idx);
    fprintf('[%d/%d] %s\n', config_idx, numel(config_grid), describe_config(config));

    try
        result_i = evaluate_config_across_datasets(datasets, config, ...
            epsilon, stage_max_iterations, optimizer_max_iterations, tol, n_bins, ...
            entropy_gamma_grid, smooth_strength, selection_metric, ...
            baseline_summary, show_dataset_progress);
        fprintf('  score=%.6f | ACC %.4f | NLL %.4f | Brier %.4f | ECE %.4f | Obj %.6f\n\n', ...
            result_i.selection_score, result_i.summary.acc_mu, ...
            result_i.summary.nll_mu, result_i.summary.brier_mu, ...
            result_i.summary.ece_mu, result_i.summary.objective_mu);
    catch ME
        result_i = struct();
        result_i.config = config;
        result_i.failed = true;
        result_i.dataset_metrics = struct([]);
        result_i.summary = empty_summary();
        result_i.selection_score = inf;
        result_i.error_message = getReport(ME, 'basic', 'hyperlinks', 'off');
        fprintf('  FAILED: %s\n\n', ME.message);
    end

    results = append_result(results, result_i);
end

[~, order] = sort([results.selection_score], 'ascend');
ranked_results = results(order);
best_result = ranked_results(1);

fprintf('\n============================================================\n');
fprintf('Best configuration\n');
fprintf('============================================================\n');
fprintf('%s\n', describe_config(best_result.config));
fprintf('Score             : %.6f\n', best_result.selection_score);
fprintf('ACC mean +/- std  : %.4f +/- %.4f\n', best_result.summary.acc_mu, best_result.summary.acc_std);
fprintf('NLL mean +/- std  : %.4f +/- %.4f\n', best_result.summary.nll_mu, best_result.summary.nll_std);
fprintf('Brier mean +/- std: %.4f +/- %.4f\n', best_result.summary.brier_mu, best_result.summary.brier_std);
fprintf('ECE mean +/- std  : %.4f +/- %.4f\n', best_result.summary.ece_mu, best_result.summary.ece_std);
fprintf('Obj mean +/- std  : %.6f +/- %.6f\n\n', best_result.summary.objective_mu, best_result.summary.objective_std);

fprintf('Top 10 configurations\n');
for rank_idx = 1:min(10, numel(ranked_results))
    result_i = ranked_results(rank_idx);
    fprintf('#%d | score %.6f | ACC %.4f | NLL %.4f | Brier %.4f | ECE %.4f | Obj %.6f\n', ...
        rank_idx, result_i.selection_score, result_i.summary.acc_mu, ...
        result_i.summary.nll_mu, result_i.summary.brier_mu, ...
        result_i.summary.ece_mu, result_i.summary.objective_mu);
    fprintf('    %s\n', describe_config(result_i.config));
end

experiment = struct();
experiment.selection_metric = selection_metric;
experiment.stage_max_iterations = stage_max_iterations;
experiment.optimizer_max_iterations = optimizer_max_iterations;
experiment.tol = tol;
experiment.epsilon = epsilon;
experiment.n_bins = n_bins;
experiment.entropy_gamma_grid = entropy_gamma_grid;
experiment.smooth_strength = smooth_strength;
experiment.baseline_summary = baseline_summary;
experiment.config_grid = config_grid;
experiment.results = results;
experiment.ranked_results = ranked_results;
experiment.best_result = best_result;

save('results_realdata_entropy_continuation.mat', 'experiment');
fprintf('\nSaved results to results_realdata_entropy_continuation.mat\n');

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function datasets = preload_datasets(dataset_files, n_bins)
    datasets = struct([]);

    for dataset_idx = 1:numel(dataset_files)
        dataset_path = fullfile(dataset_files(dataset_idx).folder, dataset_files(dataset_idx).name);
        [U, y_true, dataset_info] = load_rcmo_dataset(dataset_path);
        [~, ~, K] = size(U);

        dataset = struct();
        dataset.name = dataset_info.name;
        dataset.U = U;
        dataset.y_true = y_true;
        dataset.K = K;
        dataset.metric_mean = compute_metric_struct(squeeze(mean(U, 2)), y_true, K, n_bins);
        dataset.metric_geo = compute_metric_struct(compute_geometric_mean(U), y_true, K, n_bins);
        datasets = append_dataset(datasets, dataset);
    end
end

function config_grid = build_config_grid(method_list, preprocess_mode_list, ...
        theta_init_list, acr_start_list, B_schedule_list)
    config_grid = struct([]);

    for method_idx = 1:numel(method_list)
        method = method_list{method_idx};

        for prep_idx = 1:numel(preprocess_mode_list)
            for init_idx = 1:numel(theta_init_list)
                for schedule_idx = 1:numel(B_schedule_list)
                    if strcmp(method, 'ACR')
                        starts = acr_start_list;
                    else
                        starts = {'raw'};
                    end

                    for start_idx = 1:numel(starts)
                        config = struct();
                        config.method = method;
                        config.preprocess_mode = preprocess_mode_list{prep_idx};
                        config.theta_init = theta_init_list{init_idx};
                        config.B_schedule = B_schedule_list{schedule_idx};
                        config.acr_start = starts{start_idx};
                        config_grid = append_config(config_grid, config);
                    end
                end
            end
        end
    end
end

function result = evaluate_config_across_datasets(datasets, config, epsilon, ...
        stage_max_iterations, optimizer_max_iterations, tol, n_bins, ...
        entropy_gamma_grid, smooth_strength, selection_metric, baseline_summary, ...
        show_dataset_progress)
    metrics = struct([]);

    for dataset_idx = 1:numel(datasets)
        dataset = datasets(dataset_idx);

        if show_dataset_progress
            fprintf('  Dataset %d/%d: %s ... ', dataset_idx, numel(datasets), dataset.name);
        end

        U = preprocess_for_config(dataset.U, config.preprocess_mode, epsilon, ...
            entropy_gamma_grid, smooth_strength);
        [Q, info] = run_continuation(U, config, epsilon, ...
            stage_max_iterations, optimizer_max_iterations, tol);

        metric = compute_metric_struct(Q, dataset.y_true, dataset.K, n_bins);
        metric.objective = info.final_objective / max(1, size(U, 1) * size(U, 2));
        metric.baseline_mean = dataset.metric_mean;
        metrics = append_metric(metrics, metric);

        if show_dataset_progress
            fprintf('ACC %.4f | NLL %.4f | Brier %.4f | ECE %.4f | Obj %.6f\n', ...
                metric.acc, metric.nll, metric.brier, metric.ece, metric.objective);
        end
    end

    result = struct();
    result.config = config;
    result.failed = false;
    result.dataset_metrics = metrics;
    result.summary = summarize_metrics(metrics);
    result.selection_score = compute_selection_score(result.summary, ...
        baseline_summary.mean, selection_metric);
    result.error_message = '';
end

function [Q, info] = run_continuation(U, config, epsilon, ...
        stage_max_iterations, optimizer_max_iterations, tol)
    B_schedule = config.B_schedule;
    Q = [];
    info = struct('final_objective', inf, 'objective_history', []);

    switch config.method
        case 'CFCR'
            Theta = build_initial_theta(U, 4, B_schedule(1), config.theta_init);

            for stage_idx = 1:numel(B_schedule)
                B = B_schedule(stage_idx);
                [Q, Theta, ~, stage_info] = run_CFCR(U, ...
                    'B', B, ...
                    'epsilon', epsilon, ...
                    'max_iterations', stage_max_iterations, ...
                    'optimizer_max_iterations', optimizer_max_iterations, ...
                    'tol', tol, ...
                    'init_consensus', Q, ...
                    'init_theta', Theta, ...
                    'verbose', false);
                info = merge_stage_info(info, stage_info);
            end

        case 'ACR'
            if strcmp(config.acr_start, 'cfcr')
                init_theta_cfcr = build_initial_theta(U, 4, B_schedule(1), config.theta_init);

                for stage_idx = 1:numel(B_schedule)
                    B = B_schedule(stage_idx);
                    [Q, init_theta_cfcr] = run_CFCR(U, ...
                        'B', B, ...
                        'epsilon', epsilon, ...
                        'max_iterations', stage_max_iterations, ...
                        'optimizer_max_iterations', optimizer_max_iterations, ...
                        'tol', tol, ...
                        'init_consensus', Q, ...
                        'init_theta', init_theta_cfcr, ...
                        'verbose', false);
                end
            end

            Theta = build_initial_theta(U, 5, B_schedule(1), config.theta_init);

            for stage_idx = 1:numel(B_schedule)
                B = B_schedule(stage_idx);
                [Q, Theta, ~, stage_info] = run_ACR(U, ...
                    'B', B, ...
                    'epsilon', epsilon, ...
                    'max_iterations', stage_max_iterations, ...
                    'optimizer_max_iterations', optimizer_max_iterations, ...
                    'tol', tol, ...
                    'init_consensus', Q, ...
                    'init_theta', Theta, ...
                    'verbose', false);
                info = merge_stage_info(info, stage_info);
            end

        otherwise
            error('Unsupported method: %s', config.method);
    end
end

function info = merge_stage_info(info, stage_info)
    info.final_objective = stage_info.final_objective;
    info.objective_history = [info.objective_history; stage_info.objective_history(:)];
end

function U = preprocess_for_config(U, preprocess_mode, epsilon, gamma_grid, smooth_strength)
    U = simplex_hygiene(U, epsilon);

    switch preprocess_mode
        case 'none'
            return;

        case 'entropy_align'
            U = entropy_align_oracles(U, gamma_grid);

        case 'entropy_align_smooth'
            U = entropy_align_oracles(U, gamma_grid);
            U = adaptive_confidence_smoothing(U, smooth_strength);

        otherwise
            error('Unsupported preprocess_mode: %s', preprocess_mode);
    end

    U = simplex_hygiene(U, epsilon);
end

function U = simplex_hygiene(U, epsilon)
    U = double(U);
    U = max(U, epsilon);
    U = U ./ sum(U, 3);
end

function U_aligned = entropy_align_oracles(U, gamma_grid)
    [n, M, K] = size(U);
    mean_entropy = zeros(1, M);

    for m = 1:M
        U_m = squeeze(U(:, m, :));
        mean_entropy(m) = mean(normalized_entropy(U_m));
    end

    target_entropy = median(mean_entropy);
    U_aligned = zeros(n, M, K);

    for m = 1:M
        U_m = squeeze(U(:, m, :));
        log_U_m = log(max(U_m, 1e-12));
        best_gamma = 1;
        best_gap = inf;

        for gamma_idx = 1:numel(gamma_grid)
            gamma = gamma_grid(gamma_idx);
            candidate = softmax_rows(log_U_m * gamma);
            gap = abs(mean(normalized_entropy(candidate)) - target_entropy);

            if gap < best_gap
                best_gap = gap;
                best_gamma = gamma;
            end
        end

        U_aligned(:, m, :) = reshape(softmax_rows(log_U_m * best_gamma), n, 1, K);
    end
end

function U = adaptive_confidence_smoothing(U, smooth_strength)
    [~, ~, K] = size(U);
    max_prob = max(U, [], 3);
    excess_confidence = min(max((max_prob - 0.9) / 0.1, 0), 1);
    lambda = smooth_strength * excess_confidence;
    U = U .* (1 - lambda) + lambda / K;
    U = U ./ sum(U, 3);
end

function Theta = build_initial_theta(U, d, B, init_name)
    M = size(U, 2);

    switch init_name
        case 'zero'
            Theta = zeros(M, d);

        case 'entropy_match'
            Theta = entropy_match_theta(U, d, B);

        otherwise
            error('Unsupported theta initialization: %s', init_name);
    end
end

function Theta = entropy_match_theta(U, d, B)
    [~, M, ~] = size(U);
    eps_smooth = 1e-12;
    Q0 = hellinger_barycenter_from_tensor(U, eps_smooth);
    target_entropy = mean(normalized_entropy(Q0));
    beta_grid = linspace(-B, B, 101);
    Theta = zeros(M, d);

    for m = 1:M
        U_m = squeeze(U(:, m, :));
        log_U_m = log(max(U_m, eps_smooth));
        best_beta = 0;
        best_gap = inf;

        for beta_idx = 1:numel(beta_grid)
            beta = beta_grid(beta_idx);
            candidate = softmax_rows(log_U_m * exp(beta));
            gap = abs(mean(normalized_entropy(candidate)) - target_entropy);

            if gap < best_gap
                best_gap = gap;
                best_beta = beta;
            end
        end

        Theta(m, 1) = best_beta;
    end

    Theta = project_theta_to_constraints(Theta, B);
end

function Q = hellinger_barycenter_from_tensor(U, eps_smooth)
    n = size(U, 1);
    K = size(U, 3);
    sum_sqrt = reshape(sum(sqrt(max(U, eps_smooth)), 2), n, K);
    Q = sum_sqrt .^ 2;
    Q = normalize_rows(Q, eps_smooth);
end

function H = normalized_entropy(P)
    K = size(P, 2);
    P = max(P, 1e-12);
    H = -sum(P .* log(P), 2) / log(K);
end

function P = softmax_rows(logits)
    logits = logits - max(logits, [], 2);
    P = exp(logits);
    P = P ./ sum(P, 2);
end

function summary = summarize_metrics(metrics)
    summary = struct();
    summary.acc_mu = mean([metrics.acc]); summary.acc_std = std([metrics.acc]);
    summary.nll_mu = mean([metrics.nll]); summary.nll_std = std([metrics.nll]);
    summary.brier_mu = mean([metrics.brier]); summary.brier_std = std([metrics.brier]);
    summary.ece_mu = mean([metrics.ece]); summary.ece_std = std([metrics.ece]);
    summary.objective_mu = mean([metrics.objective]);
    summary.objective_std = std([metrics.objective]);
end

function summary = empty_summary()
    summary = struct();
    summary.acc_mu = nan; summary.acc_std = nan;
    summary.nll_mu = nan; summary.nll_std = nan;
    summary.brier_mu = nan; summary.brier_std = nan;
    summary.ece_mu = nan; summary.ece_std = nan;
    summary.objective_mu = nan; summary.objective_std = nan;
end

function score = compute_selection_score(summary, baseline, selection_metric)
    switch selection_metric
        case 'acc'
            score = -summary.acc_mu;
        case 'nll'
            score = summary.nll_mu;
        case 'brier'
            score = summary.brier_mu;
        case 'ece'
            score = summary.ece_mu;
        case 'objective'
            score = summary.objective_mu;
        case 'balanced'
            score = ...
                (baseline.acc_mu - summary.acc_mu) + ...
                relative_delta(summary.nll_mu, baseline.nll_mu) + ...
                relative_delta(summary.brier_mu, baseline.brier_mu) + ...
                relative_delta(summary.ece_mu, baseline.ece_mu);
        otherwise
            error('Unsupported selection_metric: %s', selection_metric);
    end
end

function value = relative_delta(candidate, baseline)
    value = (candidate - baseline) / max(abs(baseline), 1e-12);
end

function baseline_summary = summarize_baselines(datasets)
    mean_metrics = [datasets.metric_mean];
    geo_metrics = [datasets.metric_geo];
    baseline_summary = struct();
    baseline_summary.mean = summarize_metrics_with_default_objective(mean_metrics);
    baseline_summary.geo = summarize_metrics_with_default_objective(geo_metrics);
end

function summary = summarize_metrics_with_default_objective(metrics)
    for idx = 1:numel(metrics)
        metrics(idx).objective = nan;
    end
    summary = summarize_metrics(metrics);
end

function text = describe_config(config)
    text = sprintf('method=%s, preprocess=%s, theta_init=%s, B_schedule=%s, acr_start=%s', ...
        config.method, config.preprocess_mode, config.theta_init, ...
        mat2str(config.B_schedule), config.acr_start);
end

function [U, y_true, dataset_info] = load_rcmo_dataset(dataset_path)
    data = load(dataset_path);

    if ~isfield(data, 'P_cell') || ~iscell(data.P_cell)
        error('Dataset %s does not contain a valid P_cell.', dataset_path);
    end

    if ~isfield(data, 'gt')
        error('Dataset %s does not contain gt.', dataset_path);
    end

    raw_labels = double(data.gt(:));

    if min(raw_labels) == 0
        y_true = raw_labels + 1;
    else
        y_true = raw_labels;
    end

    n = numel(y_true);
    M = numel(data.P_cell);
    first_prob = standardize_probability_matrix(data.P_cell{1}, n, []);
    [n_check, K] = size(first_prob);
    U = zeros(n_check, M, K);
    U(:, 1, :) = reshape(first_prob, n_check, 1, K);

    for m = 2:M
        prob_m = standardize_probability_matrix(data.P_cell{m}, n_check, K);
        U(:, m, :) = reshape(prob_m, n_check, 1, K);
    end

    if any(y_true < 1) || any(y_true > K)
        error('Labels in %s are outside the valid range 1..K.', dataset_path);
    end

    [~, name, ext] = fileparts(dataset_path);
    dataset_info = struct();
    dataset_info.name = [name ext];
    dataset_info.path = dataset_path;
end

function prob = standardize_probability_matrix(prob, n_expected, K_expected)
    prob = double(prob);

    if ~ismatrix(prob)
        error('Probability matrix must be 2D.');
    end

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

function Q = compute_geometric_mean(U)
    [n, ~, K] = size(U);
    mean_log_output = reshape(mean(log(max(U, 1e-12)), 2), n, K);
    Q = exp(mean_log_output);
    Q = normalize_rows(Q, 1e-12);
end

function metric = compute_metric_struct(Q, y_true, K, n_bins)
    [~, y_pred] = max(Q, [], 2);
    metric = struct();
    metric.acc = compute_accuracy(y_pred, y_true);
    metric.nll = compute_nll(Q, y_true);
    metric.brier = compute_brier_score(Q, y_true, K);
    metric.ece = compute_ece(Q, y_true, n_bins);
end

function acc = compute_accuracy(y_pred, y_true)
    acc = sum(y_pred == y_true) / length(y_true);
end

function nll = compute_nll(Q, y_true)
    n = length(y_true);
    idx = sub2ind(size(Q), (1:n)', y_true(:));
    nll = -mean(log(max(Q(idx), 1e-15)));
end

function brier = compute_brier_score(Q, y_true, K)
    n = length(y_true);
    Y_true_onehot = zeros(n, K);
    idx = sub2ind([n, K], (1:n)', y_true(:));
    Y_true_onehot(idx) = 1;
    brier = sum((Q - Y_true_onehot) .^ 2, 'all') / n;
end

function ece = compute_ece(Q, y_true, n_bins)
    [max_probs, pred_labels] = max(Q, [], 2);
    correct = (pred_labels == y_true);
    ece = 0;

    for bin_idx = 1:n_bins
        lower_bound = (bin_idx - 1) / n_bins;
        upper_bound = bin_idx / n_bins;
        in_bin = (max_probs > lower_bound) & (max_probs <= upper_bound);

        if sum(in_bin) == 0
            continue;
        end

        acc_in_bin = sum(correct(in_bin)) / sum(in_bin);
        conf_in_bin = mean(max_probs(in_bin));
        bin_weight = sum(in_bin) / numel(y_true);
        ece = ece + bin_weight * abs(acc_in_bin - conf_in_bin);
    end
end

function X = normalize_rows(X, eps_smooth)
    row_sums = max(sum(X, 2), eps_smooth);
    X = X ./ row_sums;
end

function Theta_proj = project_theta_to_constraints(Theta, B)
    [M, d] = size(Theta);
    Theta_proj = zeros(M, d);

    for dim_idx = 1:d
        v = Theta(:, dim_idx);
        lo = min(v) - B;
        hi = max(v) + B;

        for iter = 1:80
            mid = (lo + hi) / 2;
            projected = min(max(v - mid, -B), B);
            if sum(projected) > 0
                lo = mid;
            else
                hi = mid;
            end
        end

        Theta_proj(:, dim_idx) = min(max(v - (lo + hi) / 2, -B), B);
    end
end

function array = append_config(array, item)
    if isempty(array)
        array = item;
    else
        array(end + 1) = item;
    end
end

function array = append_result(array, item)
    if isempty(array)
        array = item;
    else
        array(end + 1) = item;
    end
end

function array = append_dataset(array, item)
    if isempty(array)
        array = item;
    else
        array(end + 1) = item;
    end
end

function array = append_metric(array, item)
    if isempty(array)
        array = item;
    else
        array(end + 1) = item;
    end
end
