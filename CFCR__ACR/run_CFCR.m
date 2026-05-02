%% CFCR: Closed-form Comparability Recovery
% This function implements the Closed-form Comparability Recovery method
% as described in "When Aligned Soft Outputs Still Fail to Fuse".
%
% Input:
%   U: n x M x K tensor, soft outputs from M oracles on n instances with K classes
%      U(i,m,:) is the probability distribution for instance i from oracle m
%   varargin: optional parameters (bound B, max iterations T, smoothing epsilon)
%
% Output:
%   Q: n x K matrix, consensus distributions after CFCR
%   Theta: M x d matrix, learned oracle-specific parameters
%   rectified_outputs: n x M x K tensor, rectified oracle outputs

function [Q, Theta, rectified_outputs, info] = run_CFCR(U, varargin)

    % =====================================================================
    % 1. PARAMETER INITIALIZATION
    % =====================================================================
    [~, M, K] = size(U);

    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'B', 2.0);                  % bound on theta parameters
    addParameter(p, 'max_iterations', 50);      % maximum iterations
    addParameter(p, 'epsilon', 1e-8);           % smoothing constant
    addParameter(p, 'descriptor_dim', 4);       % descriptor dimension d
    addParameter(p, 'tol', 1e-6);               % convergence tolerance
    addParameter(p, 'verbose', true);           % display progress
    addParameter(p, 'optimizer_max_iterations', 100); % fmincon max iterations
    addParameter(p, 'init_consensus', []);      % optional initialization for Q
    addParameter(p, 'init_theta', []);          % optional initialization for Theta
    parse(p, varargin{:});

    B = p.Results.B;
    T = p.Results.max_iterations;
    eps_smooth = p.Results.epsilon;
    d = p.Results.descriptor_dim;
    tol = p.Results.tol;
    verbose = p.Results.verbose;
    optimizer_max_iterations = p.Results.optimizer_max_iterations;
    init_consensus = p.Results.init_consensus;
    init_theta = p.Results.init_theta;

    if d ~= 4
        error('CFCR requires descriptor_dim = 4.');
    end

    % =====================================================================
    % 2. COMPUTE FROZEN GEOMETRY DESCRIPTORS
    % =====================================================================
    Phi = compute_frozen_descriptors(U, K);

    % =====================================================================
    % 3. INITIALIZATION
    % =====================================================================
    Q = initialize_consensus(U, eps_smooth, init_consensus);
    Theta = initialize_theta(M, d, B, init_theta);
    log_U_eps = log(U + eps_smooth);
    F_history = nan(T, 1);

    % =====================================================================
    % 4. ALTERNATING OPTIMIZATION
    % =====================================================================
    for iter = 1:T

        Theta_prev = Theta;

        % Update each oracle independently under box constraints, then
        % project the parameter matrix onto the centering constraint.
        Theta = update_all_oracle_parameters(log_U_eps, Q, Phi, Theta, B, ...
            optimizer_max_iterations);

        [rectified_outputs, Q_new, F_curr] = ...
            compute_rectified_outputs_consensus_and_objective(log_U_eps, Theta, Phi, ...
            eps_smooth);

        F_history(iter) = F_curr;
        q_change = max(abs(Q_new(:) - Q(:)));
        theta_change = max(abs(Theta(:) - Theta_prev(:)));

        if iter == 1
            objective_change = inf;
        else
            objective_change = abs(F_history(iter) - F_history(iter - 1));
        end

        if verbose && (iter == 1 || mod(iter, 10) == 0)
            fprintf('CFCR Iteration %d: F = %.6f, dF = %.3e, dQ = %.3e, dTheta = %.3e\n', ...
                iter, F_curr, objective_change, q_change, theta_change);
        end

        Q = Q_new;

        % Stop either when the objective stabilizes or when both iterates do.
        if iter > 1 && (objective_change < tol || (q_change < tol && theta_change < tol))
            if verbose
                fprintf('CFCR converged at iteration %d\n', iter);
            end
            break;
        end
    end

    if verbose
        valid_F = F_history(~isnan(F_history));
        fprintf('CFCR completed. Final objective value: %.6f\n', valid_F(end));
    end

    valid_F = F_history(~isnan(F_history));
    info = struct();
    info.objective_history = valid_F;
    info.final_objective = valid_F(end);
    info.iterations = numel(valid_F);

end

% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

%% Compute Frozen Geometry Descriptors
function Phi = compute_frozen_descriptors(U, K)
    n = size(U, 1);
    U_safe = max(U, 1e-10);
    U_bar = mean(U, 2);
    sorted_U = sort(U, 3, 'descend');

    mean_entropy = reshape(mean(-sum(U_safe .* log(U_safe), 3) / log(K), 2), n, 1);
    mean_h2_to_avg = compute_mean_h2_to_avg(U, U_bar);
    mean_tail_mass = reshape(mean(1 - sorted_U(:, :, 1) - sorted_U(:, :, 2), 2), n, 1);

    Phi = [ones(n, 1), mean_entropy, mean_h2_to_avg, mean_tail_mass];
end

%% Initialize consensus by Hellinger barycenter of raw outputs
function Q = initialize_consensus(U, eps_smooth, init_consensus)
    n = size(U, 1);
    K = size(U, 3);

    if isempty(init_consensus)
        mean_sqrt_output = reshape(mean(sqrt(U), 2), n, K);
        Q = mean_sqrt_output .^ 2;
    else
        Q = double(init_consensus);
        if ~isequal(size(Q), [n, K])
            error('init_consensus must have size n x K.');
        end
    end

    Q = max(Q, eps_smooth);
    Q = normalize_rows(Q, eps_smooth);
end

%% Initialize theta under the paper constraints
function Theta = initialize_theta(M, d, B, init_theta)
    if isempty(init_theta)
        Theta = zeros(M, d);
        return;
    end

    Theta = double(init_theta);
    if ~isequal(size(Theta), [M, d])
        error('init_theta must have size M x descriptor_dim.');
    end

    Theta = project_theta_to_constraints(Theta, B);
end

%% Compute rectified outputs, updated consensus, and objective value
function [rectified_outputs, Q_new, F_curr] = ...
        compute_rectified_outputs_consensus_and_objective(log_U_eps, Theta, Phi, ...
        eps_smooth)
    eta = Phi * Theta';
    alpha = exp(eta);

    rectified_outputs = apply_rectifier_from_logs(log_U_eps, alpha, eps_smooth);
    Q_new = compute_consensus_from_rectified(rectified_outputs, eps_smooth);
    F_curr = compute_objective_from_rectified(rectified_outputs, Q_new);
end

%% Update Oracle Parameters (Theta)
function Theta_new = update_all_oracle_parameters(log_U_eps, Q, Phi, Theta, B, ...
        optimizer_max_iterations)
    [~, M, ~] = size(log_U_eps);
    d = size(Phi, 2);

    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'MaxIterations', optimizer_max_iterations, ...
        'SpecifyObjectiveGradient', true);

    Theta_new = Theta;
    lb = -B * ones(d, 1);
    ub =  B * ones(d, 1);

    for m = 1:M
        log_U_m = squeeze(log_U_eps(:, m, :));
        obj_func = @(theta_m) compute_objective_single_oracle(log_U_m, Q, Phi, theta_m);
        Theta_new(m, :) = fmincon(obj_func, Theta(m, :)', [], [], [], [], lb, ub, [], options)';
    end

    Theta_new = project_theta_to_constraints(Theta_new, B);
end

%% Single-oracle objective for Eq. (update_THETA)
function [F, grad] = compute_objective_single_oracle(log_U_m, Q, Phi, theta_m)
    eta = Phi * theta_m;
    alpha = exp(eta);
    rectified = softmax_single_oracle(log_U_m, alpha);

    sqrt_rectified = sqrt(max(rectified, 1e-15));
    sqrt_Q = sqrt(max(Q, 1e-15));
    diff = sqrt_rectified - sqrt_Q;
    F = 0.5 * sum(diff .^ 2, 'all');

    if nargout > 1
        grad_r = 0.5 * (1 - sqrt_Q ./ sqrt_rectified);
        mean_grad_r = sum(grad_r .* rectified, 2);
        centered_log_u = log_U_m - sum(rectified .* log_U_m, 2);
        grad_alpha = sum(rectified .* centered_log_u .* (grad_r - mean_grad_r), 2);
        grad = Phi' * (grad_alpha .* alpha);
    end
end

%% Apply rectifier using precomputed logs
function rectified = apply_rectifier_from_logs(log_U_eps, alpha, eps_smooth)
    rectified = softmax_along_classes(log_U_eps, alpha);
    rectified = max(rectified, eps_smooth);
    rectified = rectified ./ sum(rectified, 3);
end

%% Stable softmax along class dimension
function P = softmax_along_classes(log_U_eps, alpha)
    [n, M, ~] = size(log_U_eps);
    logits = log_U_eps .* reshape(alpha, n, M, 1);
    logits = logits - max(logits, [], 3);
    P = exp(logits);
    P = P ./ sum(P, 3);
end

%% Stable softmax for one oracle
function P = softmax_single_oracle(log_U_m, alpha)
    logits = log_U_m .* alpha;
    logits = logits - max(logits, [], 2);
    P = exp(logits);
    P = P ./ sum(P, 2);
end

%% Compute consensus from rectified outputs
function Q = compute_consensus_from_rectified(rectified_outputs, eps_smooth)
    n = size(rectified_outputs, 1);
    K = size(rectified_outputs, 3);

    sum_sqrt_rectified = reshape(sum(sqrt(rectified_outputs), 2), n, K);
    Q = sum_sqrt_rectified .^ 2;
    Q = normalize_rows(Q, eps_smooth);
end

%% Compute objective value from rectified outputs and consensus
function F = compute_objective_from_rectified(rectified_outputs, Q)
    n = size(rectified_outputs, 1);
    K = size(rectified_outputs, 3);

    sqrt_rectified = sqrt(rectified_outputs);
    sqrt_Q = reshape(sqrt(max(Q, 1e-15)), n, 1, K);
    diff = sqrt_rectified - sqrt_Q;
    F = 0.5 * sum(diff .^ 2, 'all');
end

%% Mean squared Hellinger distance to oracle average
function mean_h2_to_avg = compute_mean_h2_to_avg(U, U_bar)
    n = size(U, 1);
    h2 = 0.5 * sum((sqrt(U) - sqrt(U_bar)) .^ 2, 3);
    mean_h2_to_avg = reshape(mean(h2, 2), n, 1);
end

%% Normalize rows to the simplex
function X = normalize_rows(X, eps_smooth)
    row_sums = sum(X, 2);
    row_sums = max(row_sums, eps_smooth);
    X = X ./ row_sums;
end

%% Project onto theta_md in [-B, B] and sum_m theta_md = 0
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
