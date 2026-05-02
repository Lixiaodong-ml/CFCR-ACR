classdef utils
    % Shared utilities for mechanism-oriented CFCR/ACR experiments.
    % All methods are static so experiment scripts can call mech.utils.*.

    methods (Static)
        function cfg = defaultConfig()
            here = fileparts(mfilename('fullpath'));
            exp_dir = fileparts(here);
            cfg = struct();
            cfg.n = 1000;
            cfg.M = 10;
            cfg.K = 8;
            cfg.seed = 7;
            cfg.latent_temperature = 0.9;
            cfg.epsilon = 1e-8;
            cfg.max_iterations = 30;
            cfg.optimizer_max_iterations = 50;
            cfg.B = 3;
            cfg.tol = 1e-6;
            cfg.n_bins = 10;
            cfg.output_dir = fullfile(exp_dir, 'results');
            cfg.distortion_strength = 1.0;
            cfg.mismatch_strength = 0.0;
            cfg.num_repeats = 1;
            cfg.verbose = false;
        end

        function ensurePaths()
            here = fileparts(mfilename('fullpath'));
            exp_dir = fileparts(here);
            root_dir = fileparts(exp_dir);
            addpath(root_dir);
            addpath(exp_dir);
        end

        function out_dir = makeOutputDir(cfg, exp_name)
            out_dir = fullfile(cfg.output_dir, exp_name);
            if ~exist(out_dir, 'dir')
                mkdir(out_dir);
            end
        end

        function writeStructTable(rows, out_path)
            T = struct2table(rows);
            writetable(T, out_path);
        end

        function summary_rows = writeRepeatedTables(rows, out_path, group_fields)
            summary_rows = mech.utils.meanStdRows(rows, group_fields);
            mech.utils.writeStructTable(summary_rows, out_path);
        end

        function summary_rows = meanStdRows(rows, group_fields)
            if isempty(rows)
                summary_rows = rows;
                return;
            end

            all_fields = fieldnames(rows);
            group_fields = cellstr(group_fields);
            skip_fields = [{'repeat'}, group_fields(:)'];
            metric_fields = {};
            first_row = rows(1);
            for i = 1:numel(all_fields)
                field = all_fields{i};
                if any(strcmp(field, skip_fields))
                    continue;
                end
                value = first_row.(field);
                if isnumeric(value) && isscalar(value)
                    metric_fields{end + 1} = field; %#ok<AGROW>
                end
            end

            keys = strings(1, numel(rows));
            for i = 1:numel(rows)
                parts = strings(1, numel(group_fields));
                for g = 1:numel(group_fields)
                    value = rows(i).(group_fields{g});
                    if isstring(value) || ischar(value)
                        parts(g) = string(value);
                    elseif isnumeric(value) || islogical(value)
                        parts(g) = string(value);
                    else
                        parts(g) = string(evalc('disp(value)'));
                    end
                end
                keys(i) = strjoin(parts, "||");
            end
            unique_keys = unique(keys, 'stable');

            summary_rows = struct([]);
            for k = 1:numel(unique_keys)
                idx = find(keys == unique_keys(k));
                out = struct();
                for g = 1:numel(group_fields)
                    out.(group_fields{g}) = rows(idx(1)).(group_fields{g});
                end
                out.n_repeats = numel(idx);
                for m = 1:numel(metric_fields)
                    field = metric_fields{m};
                    vals = arrayfun(@(x) x.(field), rows(idx));
                    vals = vals(~isnan(vals));
                    if isempty(vals)
                        mu = NaN;
                        sigma = NaN;
                    else
                        mu = mean(vals);
                        sigma = std(vals, 0);
                    end
                    out.(field) = string(sprintf('%.6g ± %.6g', mu, sigma));
                end
                if isempty(summary_rows)
                    summary_rows = out;
                else
                    summary_rows(k) = orderfields(out, summary_rows); %#ok<AGROW>
                end
            end
        end

        function Pi = latentConsensus(n, K, temperature, eps_smooth)
            logits = randn(n, K) ./ temperature;
            Pi = mech.utils.softmaxRows(logits);
            Pi = max(Pi, eps_smooth);
            Pi = mech.utils.normalizeRows(Pi, eps_smooth);
        end

        function P = softmaxRows(logits)
            logits = logits - max(logits, [], 2);
            P = exp(logits);
            P = P ./ sum(P, 2);
        end

        function X = normalizeRows(X, eps_smooth)
            X = max(X, eps_smooth);
            X = X ./ max(sum(X, 2), eps_smooth);
        end

        function U = rectify(P, alpha, eps_smooth)
            % P can be n x K or n x M x K. alpha can be scalar, n x 1,
            % n x M, or n x M x K.
            if nargin < 3
                eps_smooth = 1e-8;
            end
            P = max(P, eps_smooth);
            if ndims(P) == 2
                if isscalar(alpha)
                    logits = log(P) .* alpha;
                else
                    logits = log(P) .* alpha(:);
                end
                U = mech.utils.softmaxRows(logits);
            else
                [n, M, ~] = size(P);
                if isscalar(alpha)
                    A = alpha;
                elseif ndims(alpha) == 2
                    A = reshape(alpha, n, M, 1);
                else
                    A = alpha;
                end
                logits = log(P) .* A;
                logits = logits - max(logits, [], 3);
                U = exp(logits);
                U = U ./ sum(U, 3);
            end
        end

        function [U, Pi, y, truth] = makeSynthetic(scenario, cfg)
            rng(cfg.seed);
            n = cfg.n;
            M = cfg.M;
            K = cfg.K;
            eps_smooth = cfg.epsilon;
            Pi = mech.utils.latentConsensus(n, K, cfg.latent_temperature, eps_smooth);
            [~, y] = max(Pi, [], 2);

            sorted_pi = sort(Pi, 2, 'descend');
            gap = sorted_pi(:, 1) - sorted_pi(:, 2);
            tail = 1 - sorted_pi(:, 1) - sorted_pi(:, 2);
            entropy = -sum(Pi .* log(max(Pi, eps_smooth)), 2) / log(K);

            oracle_axis = linspace(-1, 1, M);
            alpha = ones(n, M);
            strength = cfg.distortion_strength;

            switch lower(scenario)
                case 'global'
                    for m = 1:M
                        alpha(:, m) = exp(strength * oracle_axis(m));
                    end

                case 'boundary'
                    boundary = 1 - gap;
                    boundary = (boundary - mean(boundary)) ./ max(std(boundary), eps_smooth);
                    for m = 1:M
                        alpha(:, m) = exp(0.25 * oracle_axis(m) + 0.75 * strength * oracle_axis(m) * boundary);
                    end

                case 'tail'
                    tail_z = (tail - mean(tail)) ./ max(std(tail), eps_smooth);
                    for m = 1:M
                        alpha(:, m) = exp(0.20 * oracle_axis(m) + 0.80 * strength * oracle_axis(m) * tail_z);
                    end

                case 'mixed'
                    boundary = 1 - gap;
                    boundary = (boundary - mean(boundary)) ./ max(std(boundary), eps_smooth);
                    tail_z = (tail - mean(tail)) ./ max(std(tail), eps_smooth);
                    ent_z = (entropy - mean(entropy)) ./ max(std(entropy), eps_smooth);
                    for m = 1:M
                        alpha(:, m) = exp(0.15 * oracle_axis(m) + 0.35 * strength * oracle_axis(m) * boundary + ...
                            0.25 * strength * oracle_axis(m) * tail_z + 0.20 * strength * oracle_axis(m) * ent_z);
                    end

                case {'anisotropic', 'nonmonotone', 'approx_mismatch'}
                    for m = 1:M
                        alpha(:, m) = exp(strength * oracle_axis(m));
                    end

                otherwise
                    error('Unknown synthetic scenario: %s', scenario);
            end

            alpha = min(max(alpha, 0.20), 5.0);
            Pi3 = repmat(reshape(Pi, n, 1, K), 1, M, 1);
            U = mech.utils.rectify(Pi3, 1 ./ alpha, eps_smooth);

            if strcmpi(scenario, 'anisotropic') || strcmpi(scenario, 'approx_mismatch')
                mismatch = cfg.mismatch_strength;
                if strcmpi(scenario, 'anisotropic')
                    mismatch = max(mismatch, 0.65);
                end
                coord = ones(n, M, K);
                for m = 1:M
                    class_axis = linspace(-1, 1, K);
                    coord(:, m, :) = repmat( ...
                        reshape(exp(mismatch * oracle_axis(m) * class_axis), 1, 1, K), ...
                        n, 1, 1);
                end
                U_aniso = mech.utils.rectify(Pi3, coord ./ reshape(alpha, n, M, 1), eps_smooth);
                U = (1 - min(mismatch, 1)) * U + min(mismatch, 1) * U_aniso;
                U = U ./ sum(U, 3);
            end

            if strcmpi(scenario, 'nonmonotone')
                swap_strength = min(max(cfg.mismatch_strength, 0.35), 1.0);
                for m = 1:M
                    if mod(m, 2) == 0
                        Um = squeeze(U(:, m, :));
                        [~, ord] = sort(Um, 2, 'descend');
                        for i = 1:n
                            a = ord(i, 1);
                            b = ord(i, 2);
                            mixed_a = (1 - swap_strength) * Um(i, a) + swap_strength * Um(i, b);
                            mixed_b = (1 - swap_strength) * Um(i, b) + swap_strength * Um(i, a);
                            Um(i, a) = mixed_a;
                            Um(i, b) = mixed_b;
                        end
                        U(:, m, :) = reshape(Um, n, 1, K);
                    end
                end
            end

            truth = struct();
            truth.alpha = alpha;
            truth.gap = gap;
            truth.tail = tail;
            truth.entropy = entropy;
            truth.hard = gap <= median(gap);
            truth.high_tail = tail >= median(tail);
            truth.scenario = scenario;
        end

        function Q = consensusMean(U)
            Q = squeeze(mean(U, 2));
            Q = mech.utils.normalizeRows(Q, 1e-12);
        end

        function Q = consensusGeo(U)
            Q = squeeze(exp(mean(log(max(U, 1e-12)), 2)));
            Q = mech.utils.normalizeRows(Q, 1e-12);
        end

        function Q = consensusHellinger(U)
            n = size(U, 1);
            K = size(U, 3);
            S = reshape(sum(sqrt(max(U, 1e-12)), 2), n, K);
            Q = S .^ 2;
            Q = mech.utils.normalizeRows(Q, 1e-12);
        end

        function Q = consensusFromRectified(U, objective)
            switch lower(objective)
                case 'hellinger'
                    Q = mech.utils.consensusHellinger(U);
                case 'euclidean'
                    Q = mech.utils.consensusMean(U);
                case {'kl', 'geometric'}
                    Q = mech.utils.consensusGeo(U);
                otherwise
                    error('Unknown consensus objective: %s', objective);
            end
        end

        function results = runCoreMethods(U, Pi, y, cfg)
            mech.utils.ensurePaths();
            methods = {};
            Qs = {};

            methods{end + 1} = 'Mean';
            Qs{end + 1} = mech.utils.consensusMean(U);
            methods{end + 1} = 'GeoMean';
            Qs{end + 1} = mech.utils.consensusGeo(U);
            methods{end + 1} = 'HellingerRaw';
            Qs{end + 1} = mech.utils.consensusHellinger(U);

            try
                [Q_cfcr, ~, R_cfcr, info_cfcr] = run_CFCR(U, ...
                    'max_iterations', cfg.max_iterations, ...
                    'optimizer_max_iterations', cfg.optimizer_max_iterations, ...
                    'B', cfg.B, 'epsilon', cfg.epsilon, 'tol', cfg.tol, ...
                    'verbose', cfg.verbose);
                methods{end + 1} = 'CFCR';
                Qs{end + 1} = Q_cfcr;
            catch ME
                warning('CFCR failed: %s', ME.message);
                R_cfcr = [];
                info_cfcr = struct();
            end

            try
                [Q_acr, ~, R_acr, info_acr] = run_ACR(U, ...
                    'max_iterations', cfg.max_iterations, ...
                    'optimizer_max_iterations', cfg.optimizer_max_iterations, ...
                    'B', cfg.B, 'epsilon', cfg.epsilon, 'tol', cfg.tol, ...
                    'verbose', cfg.verbose);
                methods{end + 1} = 'ACR';
                Qs{end + 1} = Q_acr;
            catch ME
                warning('ACR failed: %s', ME.message);
                R_acr = [];
                info_acr = struct();
            end

            rows = repmat(mech.utils.scoreConsensus(methods{1}, Qs{1}, Pi, y, cfg.n_bins), 1, numel(methods));
            for j = 2:numel(methods)
                rows(j) = mech.utils.scoreConsensus(methods{j}, Qs{j}, Pi, y, cfg.n_bins);
            end

            results = struct();
            results.rows = rows;
            results.methods = methods;
            results.Qs = Qs;
            results.R_cfcr = R_cfcr;
            results.R_acr = R_acr;
            results.info_cfcr = info_cfcr;
            results.info_acr = info_acr;
        end

        function row = scoreConsensus(name, Q, Pi, y, n_bins)
            [~, pred] = max(Q, [], 2);
            row = struct();
            row.method = string(name);
            if isempty(Pi)
                row.hellinger = NaN;
                row.kl_to_pi = NaN;
            else
                row.hellinger = mean(mech.utils.hellingerRows(Q, Pi));
                row.kl_to_pi = mean(mech.utils.klRows(Pi, Q));
            end
            row.nll = mech.utils.nll(Q, y);
            row.brier = mech.utils.brier(Q, y);
            row.ece = mech.utils.ece(Q, y, n_bins);
            row.accuracy = mean(pred == y);
        end

        function h = hellingerRows(P, Q)
            h = sqrt(0.5 * sum((sqrt(max(P, 0)) - sqrt(max(Q, 0))) .^ 2, 2));
        end

        function kl = klRows(P, Q)
            P = max(P, 1e-12);
            Q = max(Q, 1e-12);
            kl = sum(P .* (log(P) - log(Q)), 2);
        end

        function val = nll(Q, y)
            idx = sub2ind(size(Q), (1:numel(y))', y(:));
            val = -mean(log(max(Q(idx), 1e-12)));
        end

        function val = brier(Q, y)
            Y = zeros(size(Q));
            idx = sub2ind(size(Q), (1:numel(y))', y(:));
            Y(idx) = 1;
            val = mean(sum((Q - Y) .^ 2, 2));
        end

        function val = ece(Q, y, n_bins)
            [conf, pred] = max(Q, [], 2);
            correct = pred == y(:);
            edges = linspace(0, 1, n_bins + 1);
            val = 0;
            n = numel(y);
            for b = 1:n_bins
                if b < n_bins
                    mask = conf >= edges(b) & conf < edges(b + 1);
                else
                    mask = conf >= edges(b) & conf <= edges(b + 1);
                end
                if any(mask)
                    val = val + nnz(mask) / n * abs(mean(correct(mask)) - mean(conf(mask)));
                end
            end
        end

        function stats = geometryStats(U)
            [n, M, K] = size(U);
            U_safe = max(U, 1e-12);
            sorted_U = sort(U, 3, 'descend');
            ent = -sum(U_safe .* log(U_safe), 3) / log(K);
            gap = sorted_U(:, :, 1) - sorted_U(:, :, 2);
            tail = 1 - sorted_U(:, :, 1) - sorted_U(:, :, 2);

            pair_h = zeros(n, 1);
            count = 0;
            for a = 1:M
                Pa = squeeze(U(:, a, :));
                for b = (a + 1):M
                    Pb = squeeze(U(:, b, :));
                    pair_h = pair_h + mech.utils.hellingerRows(Pa, Pb);
                    count = count + 1;
                end
            end
            pair_h = pair_h / max(count, 1);

            stats = struct();
            stats.entropy_mean = mean(ent, 'all');
            stats.entropy_oracle_std = mean(std(ent, 0, 2));
            stats.gap_mean = mean(gap, 'all');
            stats.gap_oracle_std = mean(std(gap, 0, 2));
            stats.tail_mean = mean(tail, 'all');
            stats.tail_oracle_std = mean(std(tail, 0, 2));
            stats.pairwise_hellinger = mean(pair_h);
        end

        function Phi = frozenDescriptors(U, mask)
            [n, ~, K] = size(U);
            U_safe = max(U, 1e-12);
            U_bar = mean(U, 2);
            sorted_U = sort(U, 3, 'descend');
            ent = reshape(mean(-sum(U_safe .* log(U_safe), 3) / log(K), 2), n, 1);
            h2 = 0.5 * sum((sqrt(U) - sqrt(U_bar)) .^ 2, 3);
            disp = reshape(mean(h2, 2), n, 1);
            tail = reshape(mean(1 - sorted_U(:, :, 1) - sorted_U(:, :, 2), 2), n, 1);
            Phi = [ones(n, 1), ent, disp, tail];
            if nargin > 1 && ~isempty(mask)
                Phi = Phi(:, logical(mask));
            end
        end

        function Phi = refinedDescriptors(U, Q, mask)
            [n, ~, K] = size(U);
            U_bar = mean(U, 2);
            h2 = 0.5 * sum((sqrt(U) - sqrt(U_bar)) .^ 2, 3);
            disp = reshape(mean(h2, 2), n, 1);
            Q_safe = max(Q, 1e-12);
            q_sorted = sort(Q, 2, 'descend');
            boundary = 1 - (q_sorted(:, 1) - q_sorted(:, 2));
            ent = -sum(Q_safe .* log(Q_safe), 2) / log(K);
            tail = 1 - q_sorted(:, 1) - q_sorted(:, 2);
            Phi = [ones(n, 1), boundary, ent, tail, disp];
            if nargin > 2 && ~isempty(mask)
                Phi = Phi(:, logical(mask));
            end
        end

        function [Q, Theta, R, info] = customRecovery(U, mode, mask, cfg, consensusObjective)
            if nargin < 5
                consensusObjective = 'hellinger';
            end
            [~, M, ~] = size(U);
            Q = mech.utils.consensusHellinger(U);
            if strcmpi(mode, 'frozen')
                Phi0 = mech.utils.frozenDescriptors(U, mask);
            else
                Phi0 = mech.utils.refinedDescriptors(U, Q, mask);
            end
            D = size(Phi0, 2);
            Theta = zeros(M, D);
            logU = log(max(U, cfg.epsilon));
            F = nan(cfg.max_iterations, 1);

            has_fmincon = exist('fmincon', 'file') == 2;
            options = [];
            if has_fmincon
                options = optimoptions('fmincon', 'Display', 'off', ...
                    'MaxIterations', cfg.optimizer_max_iterations);
            end

            for it = 1:cfg.max_iterations
                if strcmpi(mode, 'frozen')
                    Phi = Phi0;
                else
                    Phi = mech.utils.refinedDescriptors(U, Q, mask);
                end

                for m = 1:M
                    x0 = Theta(m, :)';
                    obj = @(x) mech.utils.singleOracleObjective(squeeze(logU(:, m, :)), Q, Phi, x);
                    if has_fmincon
                        Theta(m, :) = fmincon(obj, x0, [], [], [], [], ...
                            -cfg.B * ones(D, 1), cfg.B * ones(D, 1), [], options)';
                    else
                        Theta(m, :) = fminsearch(obj, x0)';
                        Theta(m, :) = min(max(Theta(m, :), -cfg.B), cfg.B);
                    end
                end
                Theta = mech.utils.projectTheta(Theta, cfg.B);
                alpha = exp(Phi * Theta');
                R = mech.utils.rectify(U, alpha, cfg.epsilon);
                Q_new = mech.utils.consensusFromRectified(R, consensusObjective);
                F(it) = mean(mech.utils.hellingerRows(Q_new, Q));
                if max(abs(Q_new(:) - Q(:))) < cfg.tol
                    Q = Q_new;
                    break;
                end
                Q = Q_new;
            end
            info = struct();
            info.delta_history = F(~isnan(F));
        end

        function val = singleOracleObjective(logUm, Q, Phi, theta)
            alpha = exp(Phi * theta);
            P = mech.utils.softmaxRows(logUm .* alpha);
            val = 0.5 * sum((sqrt(max(P, 1e-12)) - sqrt(max(Q, 1e-12))) .^ 2, 'all');
        end

        function Theta = projectTheta(Theta, B)
            [~, D] = size(Theta);
            for d = 1:D
                v = Theta(:, d);
                lo = min(v) - B;
                hi = max(v) + B;
                for it = 1:80
                    mid = (lo + hi) / 2;
                    p = min(max(v - mid, -B), B);
                    if sum(p) > 0
                        lo = mid;
                    else
                        hi = mid;
                    end
                end
                Theta(:, d) = min(max(v - (lo + hi) / 2, -B), B);
            end
        end

        function [Q, R] = coordinateWiseRecovery(U, cfg)
            [n, M, K] = size(U);
            Q = mech.utils.consensusHellinger(U);
            logU = log(max(U, cfg.epsilon));
            A = ones(M, K);
            has_fmincon = exist('fmincon', 'file') == 2;
            options = [];
            if has_fmincon
                options = optimoptions('fmincon', 'Display', 'off', ...
                    'MaxIterations', cfg.optimizer_max_iterations);
            end

            for it = 1:cfg.max_iterations
                for m = 1:M
                    x0 = log(A(m, :))';
                    obj = @(x) mech.utils.coordObjective(squeeze(logU(:, m, :)), Q, exp(x));
                    if has_fmincon
                        x = fmincon(obj, x0, [], [], [], [], ...
                            -cfg.B * ones(K, 1), cfg.B * ones(K, 1), [], options);
                    else
                        x = fminsearch(obj, x0);
                        x = min(max(x, -cfg.B), cfg.B);
                    end
                    A(m, :) = exp(x)';
                end
                A3 = repmat(reshape(A, 1, M, K), n, 1, 1);
                R = mech.utils.rectify(U, A3, cfg.epsilon);
                Q_new = mech.utils.consensusHellinger(R);
                if max(abs(Q_new(:) - Q(:))) < cfg.tol
                    Q = Q_new;
                    break;
                end
                Q = Q_new;
            end
            A3 = repmat(reshape(A, 1, M, K), n, 1, 1);
            R = mech.utils.rectify(U, A3, cfg.epsilon);
        end

        function val = coordObjective(logUm, Q, alphaK)
            P = mech.utils.softmaxRows(logUm .* reshape(alphaK, 1, []));
            val = 0.5 * sum((sqrt(max(P, 1e-12)) - sqrt(max(Q, 1e-12))) .^ 2, 'all');
        end

        function rate = orderReversalRate(U_before, U_after)
            [n, M, ~] = size(U_before);
            reversals = 0;
            total = 0;
            for i = 1:n
                for m = 1:M
                    a = squeeze(U_before(i, m, :));
                    b = squeeze(U_after(i, m, :));
                    [~, ord] = sort(a, 'descend');
                    for r = 1:(numel(ord) - 1)
                        for s = (r + 1):numel(ord)
                            total = total + 1;
                            if b(ord(r)) < b(ord(s))
                                reversals = reversals + 1;
                            end
                        end
                    end
                end
            end
            rate = reversals / max(total, 1);
        end

        function T = subsetRows(base_name, Qs, names, Pi, y, mask, n_bins)
            idx = find(mask);
            first = mech.utils.scoreConsensus(names{1}, Qs{1}(idx, :), Pi(idx, :), y(idx), n_bins);
            first.subset = string(base_name);
            rows = repmat(first, 1, numel(names));
            for j = 2:numel(names)
                r = mech.utils.scoreConsensus(names{j}, Qs{j}(idx, :), Pi(idx, :), y(idx), n_bins);
                r.subset = string(base_name);
                rows(j) = r;
            end
            T = rows;
        end
    end
end
