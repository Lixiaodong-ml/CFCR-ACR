%% Real-data experiment script: paper-style CFCR vs ACR on RCMO datasets
% This script keeps only the core CFCR / ACR pipeline and core method
% hyperparameters. It intentionally does NOT use any of the later
% engineering enhancements such as:
%   - UAWF warm start
%   - oracle weighting
%   - theta regularization
%   - custom alpha clipping ranges
%   - random restarts
%
% Intended use:
%   Use this file when you want a cleaner "paper-style" evaluation setup
%   that is closer to the core method description.

clearvars;
close all;
clc;

% =========================================================================
% SETUP
% =========================================================================

input_dir = fullfile('RCMO_datasets', 'RCMO_input');
dataset_files = dir(fullfile(input_dir, '*.mat'));

if isempty(dataset_files)
    error('No .mat files found under %s', input_dir);
end

% Core method hyperparameters only
max_iterations = 50;
B = 500;
epsilon = 1e-8;
tol = 1e-6;
n_bins = 10;

fprintf('\n============================================================\n');
fprintf('Paper-style CFCR vs ACR on real RCMO datasets\n');
fprintf('Input directory: %s\n', input_dir);
fprintf('Datasets found: %d\n', numel(dataset_files));
fprintf('Core parameters only: B=%.2f, max_iterations=%d, epsilon=%.1e, tol=%.1e\n', ...
    B, max_iterations, epsilon, tol);
fprintf('============================================================\n\n');

% Storage for results from each dataset
results_all = struct();
results_all.acc_mean = [];
results_all.acc_geo = [];
results_all.acc_cfcr = [];
results_all.acc_acr = [];

results_all.nll_mean = [];
results_all.nll_geo = [];
results_all.nll_cfcr = [];
results_all.nll_acr = [];

results_all.brier_mean = [];
results_all.brier_geo = [];
results_all.brier_cfcr = [];
results_all.brier_acr = [];

results_all.ece_mean = [];
results_all.ece_geo = [];
results_all.ece_cfcr = [];
results_all.ece_acr = [];

dataset_results = struct([]);

% =========================================================================
% RUN EXPERIMENTS ON EACH REAL DATASET
% =========================================================================

for dataset_idx = 1:numel(dataset_files)
    dataset_path = fullfile(dataset_files(dataset_idx).folder, dataset_files(dataset_idx).name);
    [U, y_true, dataset_info] = load_rcmo_dataset(dataset_path);
    [n, M, K] = size(U);

    fprintf('>>> Dataset %d / %d: %s\n', dataset_idx, numel(dataset_files), dataset_info.name);
    fprintf('    n = %d, M = %d, K = %d\n', n, M, K);

    % ---------------------------------------------------------------------
    % Baseline methods
    % ---------------------------------------------------------------------
    Q_mean = squeeze(mean(U, 2));
    Q_geo = compute_geometric_mean(U);

    % ---------------------------------------------------------------------
    % Paper-style proposed methods
    % ---------------------------------------------------------------------
    [Q_cfcr, Theta_cfcr, ~] = run_CFCR(U, ...
        'max_iterations', max_iterations, ...
        'B', B, ...
        'epsilon', epsilon, ...
        'tol', tol, ...
        'verbose', false);

    [Q_acr, Theta_acr, ~] = run_ACR(U, ...
        'max_iterations', max_iterations, ...
        'B', B, ...
        'epsilon', epsilon, ...
        'tol', tol, ...
        'verbose', false);

    % ---------------------------------------------------------------------
    % Metrics
    % ---------------------------------------------------------------------
    [~, y_pred_mean] = max(Q_mean, [], 2);
    [~, y_pred_geo] = max(Q_geo, [], 2);
    [~, y_pred_cfcr] = max(Q_cfcr, [], 2);
    [~, y_pred_acr] = max(Q_acr, [], 2);

    acc_mean_dataset = compute_accuracy(y_pred_mean, y_true);
    acc_geo_dataset = compute_accuracy(y_pred_geo, y_true);
    acc_cfcr_dataset = compute_accuracy(y_pred_cfcr, y_true);
    acc_acr_dataset = compute_accuracy(y_pred_acr, y_true);

    nll_mean_dataset = compute_nll(Q_mean, y_true);
    nll_geo_dataset = compute_nll(Q_geo, y_true);
    nll_cfcr_dataset = compute_nll(Q_cfcr, y_true);
    nll_acr_dataset = compute_nll(Q_acr, y_true);

    brier_mean_dataset = compute_brier_score(Q_mean, y_true, K);
    brier_geo_dataset = compute_brier_score(Q_geo, y_true, K);
    brier_cfcr_dataset = compute_brier_score(Q_cfcr, y_true, K);
    brier_acr_dataset = compute_brier_score(Q_acr, y_true, K);

    ece_mean_dataset = compute_ece(Q_mean, y_true, n_bins);
    ece_geo_dataset = compute_ece(Q_geo, y_true, n_bins);
    ece_cfcr_dataset = compute_ece(Q_cfcr, y_true, n_bins);
    ece_acr_dataset = compute_ece(Q_acr, y_true, n_bins);

    % Store global summary arrays
    results_all.acc_mean = [results_all.acc_mean; acc_mean_dataset];
    results_all.acc_geo = [results_all.acc_geo; acc_geo_dataset];
    results_all.acc_cfcr = [results_all.acc_cfcr; acc_cfcr_dataset];
    results_all.acc_acr = [results_all.acc_acr; acc_acr_dataset];

    results_all.nll_mean = [results_all.nll_mean; nll_mean_dataset];
    results_all.nll_geo = [results_all.nll_geo; nll_geo_dataset];
    results_all.nll_cfcr = [results_all.nll_cfcr; nll_cfcr_dataset];
    results_all.nll_acr = [results_all.nll_acr; nll_acr_dataset];

    results_all.brier_mean = [results_all.brier_mean; brier_mean_dataset];
    results_all.brier_geo = [results_all.brier_geo; brier_geo_dataset];
    results_all.brier_cfcr = [results_all.brier_cfcr; brier_cfcr_dataset];
    results_all.brier_acr = [results_all.brier_acr; brier_acr_dataset];

    results_all.ece_mean = [results_all.ece_mean; ece_mean_dataset];
    results_all.ece_geo = [results_all.ece_geo; ece_geo_dataset];
    results_all.ece_cfcr = [results_all.ece_cfcr; ece_cfcr_dataset];
    results_all.ece_acr = [results_all.ece_acr; ece_acr_dataset];

    % Store per-dataset details
    dataset_results(dataset_idx).name = dataset_info.name;
    dataset_results(dataset_idx).path = dataset_path;
    dataset_results(dataset_idx).n = n;
    dataset_results(dataset_idx).M = M;
    dataset_results(dataset_idx).K = K;
    dataset_results(dataset_idx).model_names = dataset_info.model_names;
    dataset_results(dataset_idx).class_names = dataset_info.class_names;
    dataset_results(dataset_idx).theta_cfcr = Theta_cfcr;
    dataset_results(dataset_idx).theta_acr = Theta_acr;
    dataset_results(dataset_idx).acc_mean = acc_mean_dataset;
    dataset_results(dataset_idx).acc_geo = acc_geo_dataset;
    dataset_results(dataset_idx).acc_cfcr = acc_cfcr_dataset;
    dataset_results(dataset_idx).acc_acr = acc_acr_dataset;
    dataset_results(dataset_idx).nll_mean = nll_mean_dataset;
    dataset_results(dataset_idx).nll_geo = nll_geo_dataset;
    dataset_results(dataset_idx).nll_cfcr = nll_cfcr_dataset;
    dataset_results(dataset_idx).nll_acr = nll_acr_dataset;
    dataset_results(dataset_idx).brier_mean = brier_mean_dataset;
    dataset_results(dataset_idx).brier_geo = brier_geo_dataset;
    dataset_results(dataset_idx).brier_cfcr = brier_cfcr_dataset;
    dataset_results(dataset_idx).brier_acr = brier_acr_dataset;
    dataset_results(dataset_idx).ece_mean = ece_mean_dataset;
    dataset_results(dataset_idx).ece_geo = ece_geo_dataset;
    dataset_results(dataset_idx).ece_cfcr = ece_cfcr_dataset;
    dataset_results(dataset_idx).ece_acr = ece_acr_dataset;

    fprintf('    ACC   Mean=%.4f Geo=%.4f CFCR=%.4f ACR=%.4f\n', ...
        acc_mean_dataset, acc_geo_dataset, acc_cfcr_dataset, acc_acr_dataset);
    fprintf('    NLL   Mean=%.4f Geo=%.4f CFCR=%.4f ACR=%.4f\n', ...
        nll_mean_dataset, nll_geo_dataset, nll_cfcr_dataset, nll_acr_dataset);
    fprintf('    Brier Mean=%.4f Geo=%.4f CFCR=%.4f ACR=%.4f\n', ...
        brier_mean_dataset, brier_geo_dataset, brier_cfcr_dataset, brier_acr_dataset);
    fprintf('    ECE   Mean=%.4f Geo=%.4f CFCR=%.4f ACR=%.4f\n\n', ...
        ece_mean_dataset, ece_geo_dataset, ece_cfcr_dataset, ece_acr_dataset);
end

fprintf('All paper-style real-data experiments finished.\n\n');

% =========================================================================
% COMPUTE SUMMARY STATISTICS ACROSS DATASETS
% =========================================================================

acc_mean_mu = mean(results_all.acc_mean);
acc_mean_std = std(results_all.acc_mean);
acc_geo_mu = mean(results_all.acc_geo);
acc_geo_std = std(results_all.acc_geo);
acc_cfcr_mu = mean(results_all.acc_cfcr);
acc_cfcr_std = std(results_all.acc_cfcr);
acc_acr_mu = mean(results_all.acc_acr);
acc_acr_std = std(results_all.acc_acr);

nll_mean_mu = mean(results_all.nll_mean);
nll_mean_std = std(results_all.nll_mean);
nll_geo_mu = mean(results_all.nll_geo);
nll_geo_std = std(results_all.nll_geo);
nll_cfcr_mu = mean(results_all.nll_cfcr);
nll_cfcr_std = std(results_all.nll_cfcr);
nll_acr_mu = mean(results_all.nll_acr);
nll_acr_std = std(results_all.nll_acr);

brier_mean_mu = mean(results_all.brier_mean);
brier_mean_std = std(results_all.brier_mean);
brier_geo_mu = mean(results_all.brier_geo);
brier_geo_std = std(results_all.brier_geo);
brier_cfcr_mu = mean(results_all.brier_cfcr);
brier_cfcr_std = std(results_all.brier_cfcr);
brier_acr_mu = mean(results_all.brier_acr);
brier_acr_std = std(results_all.brier_acr);

ece_mean_mu = mean(results_all.ece_mean);
ece_mean_std = std(results_all.ece_mean);
ece_geo_mu = mean(results_all.ece_geo);
ece_geo_std = std(results_all.ece_geo);
ece_cfcr_mu = mean(results_all.ece_cfcr);
ece_cfcr_std = std(results_all.ece_cfcr);
ece_acr_mu = mean(results_all.ece_acr);
ece_acr_std = std(results_all.ece_acr);

fprintf('============================================================\n');
fprintf('Summary across %d datasets (mean +/- std)\n', numel(dataset_files));
fprintf('============================================================\n');
fprintf('Arithmetic Mean : ACC %.4f +/- %.4f | NLL %.4f +/- %.4f | Brier %.4f +/- %.4f | ECE %.4f +/- %.4f\n', ...
    acc_mean_mu, acc_mean_std, nll_mean_mu, nll_mean_std, brier_mean_mu, brier_mean_std, ece_mean_mu, ece_mean_std);
fprintf('Geometric Mean  : ACC %.4f +/- %.4f | NLL %.4f +/- %.4f | Brier %.4f +/- %.4f | ECE %.4f +/- %.4f\n', ...
    acc_geo_mu, acc_geo_std, nll_geo_mu, nll_geo_std, brier_geo_mu, brier_geo_std, ece_geo_mu, ece_geo_std);
fprintf('CFCR (Paper)    : ACC %.4f +/- %.4f | NLL %.4f +/- %.4f | Brier %.4f +/- %.4f | ECE %.4f +/- %.4f\n', ...
    acc_cfcr_mu, acc_cfcr_std, nll_cfcr_mu, nll_cfcr_std, brier_cfcr_mu, brier_cfcr_std, ece_cfcr_mu, ece_cfcr_std);
fprintf('ACR (Paper)     : ACC %.4f +/- %.4f | NLL %.4f +/- %.4f | Brier %.4f +/- %.4f | ECE %.4f +/- %.4f\n\n', ...
    acc_acr_mu, acc_acr_std, nll_acr_mu, nll_acr_std, brier_acr_mu, brier_acr_std, ece_acr_mu, ece_acr_std);

results_final = struct();
results_final.dataset_results = dataset_results;
results_final.results_all = results_all;
results_final.dataset_names = {dataset_results.name};
results_final.input_dir = input_dir;
results_final.max_iterations = max_iterations;
results_final.B = B;
results_final.epsilon = epsilon;
results_final.tol = tol;
results_final.n_bins = n_bins;
results_final.mode = 'paper_style_core_only';

results_final.acc_mean_mu = acc_mean_mu; results_final.acc_mean_std = acc_mean_std;
results_final.acc_geo_mu = acc_geo_mu; results_final.acc_geo_std = acc_geo_std;
results_final.acc_cfcr_mu = acc_cfcr_mu; results_final.acc_cfcr_std = acc_cfcr_std;
results_final.acc_acr_mu = acc_acr_mu; results_final.acc_acr_std = acc_acr_std;

results_final.nll_mean_mu = nll_mean_mu; results_final.nll_mean_std = nll_mean_std;
results_final.nll_geo_mu = nll_geo_mu; results_final.nll_geo_std = nll_geo_std;
results_final.nll_cfcr_mu = nll_cfcr_mu; results_final.nll_cfcr_std = nll_cfcr_std;
results_final.nll_acr_mu = nll_acr_mu; results_final.nll_acr_std = nll_acr_std;

results_final.brier_mean_mu = brier_mean_mu; results_final.brier_mean_std = brier_mean_std;
results_final.brier_geo_mu = brier_geo_mu; results_final.brier_geo_std = brier_geo_std;
results_final.brier_cfcr_mu = brier_cfcr_mu; results_final.brier_cfcr_std = brier_cfcr_std;
results_final.brier_acr_mu = brier_acr_mu; results_final.brier_acr_std = brier_acr_std;

results_final.ece_mean_mu = ece_mean_mu; results_final.ece_mean_std = ece_mean_std;
results_final.ece_geo_mu = ece_geo_mu; results_final.ece_geo_std = ece_geo_std;
results_final.ece_cfcr_mu = ece_cfcr_mu; results_final.ece_cfcr_std = ece_cfcr_std;
results_final.ece_acr_mu = ece_acr_mu; results_final.ece_acr_std = ece_acr_std;

save('results_realdata_paper.mat', 'results_final');
fprintf('Saved results to results_realdata_paper.mat\n');

% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

function [U, y_true, dataset_info] = load_rcmo_dataset(dataset_path)
    data = load(dataset_path);

    if ~isfield(data, 'P_cell') || ~iscell(data.P_cell)
        error('Dataset %s does not contain a valid P_cell.', dataset_path);
    end

    if ~isfield(data, 'gt')
        error('Dataset %s does not contain gt.', dataset_path);
    end

    raw_labels = double(data.gt(:));
    if isempty(raw_labels)
        error('Dataset %s has empty gt.', dataset_path);
    end

    if min(raw_labels) == 0
        y_true = raw_labels + 1;
    else
        y_true = raw_labels;
    end

    n = numel(y_true);
    M = numel(data.P_cell);
    if M == 0
        error('Dataset %s has empty P_cell.', dataset_path);
    end

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

    [~, dataset_name, ~] = fileparts(dataset_path);
    dataset_info = struct();
    dataset_info.name = dataset_name;
    dataset_info.model_names = {};
    dataset_info.class_names = {};

    if isfield(data, 'model_names')
        dataset_info.model_names = data.model_names;
    end

    if isfield(data, 'class_names')
        dataset_info.class_names = data.class_names;
    end
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
    row_sums = sum(prob, 2);

    if any(row_sums <= 0)
        error('Probability matrix contains non-positive row sums.');
    end

    prob = prob ./ row_sums;
end

function Q = compute_geometric_mean(U)
    U_log = log(U + 1e-10);
    Q = exp(squeeze(mean(U_log, 2)));
    Q = Q ./ sum(Q, 2);
end

function acc = compute_accuracy(y_pred, y_true)
    y_pred = y_pred(:);
    y_true = y_true(:);
    acc = sum(y_pred == y_true) / length(y_true);
end

function nll = compute_nll(Q, y_true)
    n = size(Q, 1);
    nll = 0;
    for i = 1:n
        true_class = y_true(i);
        prob = max(Q(i, true_class), 1e-15);
        nll = nll - log(prob);
    end
    nll = nll / n;
end

function brier = compute_brier_score(Q, y_true, K)
    n = size(Q, 1);
    Y_true_onehot = zeros(n, K);
    for i = 1:n
        Y_true_onehot(i, y_true(i)) = 1;
    end
    brier = sum((Q - Y_true_onehot) .^ 2, 'all') / n;
end

function ece = compute_ece(Q, y_true, n_bins)
    n = size(Q, 1);
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
        bin_weight = sum(in_bin) / n;
        ece = ece + bin_weight * abs(acc_in_bin - conf_in_bin);
    end
end
