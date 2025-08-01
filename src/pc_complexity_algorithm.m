clc; clear; close; setup;

[transmit.antenna, reflect.antenna, receive.antenna] = deal(4, [16, 64, 256], 4);
[channel.rank, reflect.bond] = deal(min(transmit.antenna, receive.antenna), 16);
[channel.pathloss.direct, channel.pathloss.forward, channel.pathloss.backward] = deal(db2pow(-65), db2pow(-54), db2pow(-46));
channel.weight = ones(channel.rank, 1);
[number.antenna, number.realization, flag.direct] = deal(length(reflect.antenna), 1e3, true);

for r = 1 : number.realization
	channel.direct = flag.direct * sqrt(channel.pathloss.direct) * fading_nlos(receive.antenna, transmit.antenna);
	for a = 1 : number.antenna
		channel.forward = sqrt(channel.pathloss.forward) * fading_nlos(reflect.antenna(a), transmit.antenna);
		channel.backward = sqrt(channel.pathloss.backward) * fading_nlos(receive.antenna, reflect.antenna(a));

		tic;
		[objective.manopt(a, r), counter.manopt(a, r)] = scatter_singular_manopt(channel.direct, channel.forward, channel.backward, channel.weight, reflect.bond);
		timer.manopt(a, r) = toc;

		tic;
		[objective.geodesic(a, r), counter.geodesic(a, r)] = scatter_singular_geodesic_optimized(channel.direct, channel.forward, channel.backward, channel.weight, reflect.bond);
		timer.geodesic(a, r) = toc;
	end
end

[objective.manopt, counter.manopt, timer.manopt] = deal(mean(objective.manopt, 2), mean(counter.manopt, 2), mean(timer.manopt, 2));
[objective.geodesic, counter.geodesic, timer.geodesic] = deal(mean(objective.geodesic, 2), mean(counter.geodesic, 2), mean(timer.geodesic, 2));

save('data/pc_complexity_algorithm.mat');

fprintf('Objective: %.4g, %.4g\n', [objective.manopt, objective.geodesic]');
fprintf('Counter:   %.4g, %.4g\n', [counter.manopt, counter.geodesic]');
fprintf('Timer:     %.4g, %.4g\n', [timer.manopt, timer.geodesic]');



function [G_e] = gradient_euclidean(H, H_f, H_b, rho)
	[U, ~, V] = svd(H, 'econ');
	G_e = H_b' * U * diag(rho) * V' * H_f';
end

function [Theta] = mask(Theta, L)
	G = length(Theta) / L;
	Theta = Theta .* kron(eye(G), ones(L));
end

function [D] = direction_conjugate_optimized(G_r, G_r_old, D_old, counter, cycle)
    % Periodic conjugate direction restart by steepest ascent
    if mod(counter, cycle) == 0
        gamma = 0;
    else
        % Use element-wise operations instead of trace(A*B') for speed
        denom = sum(sum(G_r_old .* G_r_old)); % == trace(G_r_old * G_r_old')
        if denom > 0
            numer = sum(sum((G_r - G_r_old) .* G_r)); % == trace((G_r - G_r_old) * G_r')
            gamma = numer / denom;
        else
            gamma = 0;
        end
        % If the conjugate direction is not ascent, restart
        check = real(sum(sum((G_r + gamma * D_old) .* G_r))); % == real(trace((G_r + gamma*D_old)' * G_r))
        if check < 0
            gamma = 0;
        end
    end
    D = G_r + gamma * D_old;
end

%% * Manopt
function [J, r, Theta_opt] = scatter_singular_manopt(H_d, H_f, H_b, rho, L)
    G = size(H_f, 1) / L;
    problem.M = stiefelcomplexfactory(L, L, G);
    problem.cost = @(Theta_3d) -cost_function(Theta_3d, H_d, H_f, H_b, rho);
    problem.egrad = @(Theta_3d) -egrad_function(Theta_3d, H_d, H_f, H_b, rho);
    % checkgradient(problem);
	options.verbosity = 0;
	options.stopfun = @mystopfun;
    [Theta_opt, J, info] = conjugategradient(problem, [], options);
	J = -J;
    r = numel([info.iter]);
end

function stopnow = mystopfun(problem, x, info, last)
	stopnow = (last >= 2 && abs(info(last).cost - info(last-1).cost) / abs(info(last-1).cost) <= 1e-4);
end

function J = cost_function(Theta_3d, H_d, H_f, H_b, rho)
	[L, ~, G] = size(Theta_3d);
    Theta = threed_to_blkdiag(Theta_3d);
    H = channel_aggregate(H_d, H_f, H_b, Theta);
    J = rho' * svd(H, 'econ');
end

function G_e_3d = egrad_function(Theta_3d, H_d, H_f, H_b, rho)
	[L, ~, G] = size(Theta_3d);
    Theta = threed_to_blkdiag(Theta_3d);
    H = channel_aggregate(H_d, H_f, H_b, Theta);
    G_e = gradient_euclidean(H, H_f, H_b, rho);
    G_e_3d = blkdiag_to_threed(G_e, L);
end

function Theta = threed_to_blkdiag(Theta_3d)
	[L, ~, G] = size(Theta_3d);
    Theta = zeros(G * L);
    for g = 1 : G
        S = (g - 1) * L + 1 : g * L;
        Theta(S, S) = Theta_3d(:, :, g);
    end
end

function Theta_3d = blkdiag_to_threed(Theta, L)
    G = size(Theta, 1) / L;
    Theta_3d = zeros(L, L, G);
    for g = 1 : G
        S = (g - 1) * L + 1 : g * L;
        Theta_3d(:, :, g) = Theta(S, S);
    end
end

%% * Geodesic optimized
function [J, r, Theta] = scatter_singular_geodesic_optimized(H_d, H_f, H_b, rho, L)
    N = size(H_f, 1);
    G = N / L;
    Theta = eye(N);
    H = channel_aggregate(H_d, H_f, H_b, Theta);
	[G_r, D] = deal(zeros(size(Theta)));
    [iter.converge, iter.tolerance, iter.counter, iter.cycle, iter.J] = deal(false, 1e-4, 0, N^2, rho' * svd(H, 'econ'));
    while ~iter.converge
        [iter.G_r, iter.D] = deal(G_r, D);
        G_e = mask(gradient_euclidean(H, H_f, H_b, rho), L);
        G_r = gradient_riemannian(Theta, G_e);
        D = direction_conjugate_optimized(G_r, iter.G_r, iter.D, iter.counter, iter.cycle);
        [Theta] = step_armijo_geodesic_optimized(H_d, H_f, H_b, rho, Theta, D, L, G);
        H = channel_aggregate(H_d, H_f, H_b, Theta);
        J = rho' * svd(H, 'econ');
        iter.converge = (abs(J - iter.J) / abs(iter.J) <= iter.tolerance);
        iter.J = J;
        iter.counter = iter.counter + 1;
    end
    r = iter.counter;
end

function [Theta] = step_armijo_geodesic_optimized(H_d, H_f, H_b, rho, Theta, D, L, G)
    % Pre-calculate the full H matrix once.
    H = channel_aggregate(H_d, H_f, H_b, Theta);

    for g = 1:G
        S = (g - 1) * L + 1 : g * L; % Indices for block g

        Theta_g = Theta(S, S);
        D_g     = D(S, S);

        % Precompute these for efficiency
        Hb_g = H_b(:, S);   % [M x L]
        Hf_g = H_f(S, :);   % [L x N]

        H_base = H - Hb_g * Theta_g * Hf_g;

        % Function handle for the local cost: only the variable part changes
        % SVD is still the bottleneck, but all fixed terms are precomputed
        fun_g = @(theta_block) rho' * svd(H_base + Hb_g * theta_block * Hf_g, 'econ');

        % Cost at the current point
        O_g = fun_g(Theta_g);

        % Pre-calculate gradient norm term
        grad_norm_term = 0.5 * trace(D_g * D_g');

        % Eigendecomposition (only once per block)
        [V, d_eig] = eig(D_g, 'vector');
        % d_eig = diag(Dmat);

        mu_g = 1e2;
        exp_d = exp(mu_g * d_eig);
        T_g = V * diag(exp_d) * V';

        % Undershoot: double mu_g until Armijo numerator is too small
        f_trial = fun_g(T_g^2 * Theta_g); % Only compute once per loop
        while (f_trial - O_g) >= (mu_g * grad_norm_term)
            mu_g = mu_g * 2;
            exp_d = exp(mu_g * d_eig);
            T_g = V * diag(exp_d) * V';
            f_trial = fun_g(T_g^2 * Theta_g);
        end

        % Overshoot: halve mu_g until Armijo condition satisfied
        armijo_check = 0.5 * mu_g * grad_norm_term;
        f_trial = fun_g(T_g * Theta_g); % Only compute once per loop
        while (f_trial - O_g) < armijo_check && (mu_g >= 1e-4)
            mu_g = mu_g * 0.5;
            armijo_check = armijo_check * 0.5;
            exp_d = exp(mu_g * d_eig);
            T_g = V * diag(exp_d) * V';
            f_trial = fun_g(T_g * Theta_g);
        end

        % Update the block and H for the next block
        Theta(S, S) = T_g * Theta_g;
        H = H_base + Hb_g * Theta(S, S) * Hf_g;
    end
end


%% * Nongeodesic optimized
% function [J, r, Theta] = scatter_singular_nongeodesic_optimized(H_d, H_f, H_b, rho, L)
%     N = size(H_f, 1);
%     G = N / L;
%     Theta = eye(N);
%     H = channel_aggregate(H_d, H_f, H_b, Theta);
% 	[G_r, D] = deal(zeros(size(Theta)));
%     [iter.converge, iter.tolerance, iter.counter, iter.cycle, iter.J] = deal(false, 1e-4, 0, N^2, rho' * svd(H, 'econ'));
%     while ~iter.converge
%         [iter.G_r, iter.D] = deal(G_r, D);
%         G_e = mask(gradient_euclidean(H, H_f, H_b, rho), L);
%         G_r = gradient_riemannian_nongeodesic(Theta, G_e);
%         D = direction_conjugate_optimized(G_r, iter.G_r, iter.D, iter.counter, iter.cycle);
%         [Theta] = step_armijo_nongeodesic_optimized(H_d, H_f, H_b, rho, Theta, D, L, G);
%         H = channel_aggregate(H_d, H_f, H_b, Theta);
%         J = rho' * svd(H, 'econ');
%         iter.converge = (abs(J - iter.J) / abs(iter.J) <= iter.tolerance);
%         iter.J = J;
%         iter.counter = iter.counter + 1;
%     end
%     r = iter.counter;
% end
%
% function [Theta] = step_armijo_nongeodesic_optimized(H_d, H_f, H_b, rho, Theta, D, L, G)
%     H = channel_aggregate(H_d, H_f, H_b, Theta);
%
%     for g = 1:G
%         S = (g - 1) * L + 1 : g * L;
%
%         Theta_g = Theta(S, S);
%         D_g     = D(S, S);
%
%         Hb_g = H_b(:, S);
%         Hf_g = H_f(S, :);
%
%         H_base = H - Hb_g * Theta_g * Hf_g;
%
%         fun_g = @(theta_block) rho' * svd(H_base + Hb_g * theta_block * Hf_g, 'econ');
%         O_g = fun_g(Theta_g);
%         grad_norm_term = 0.5 * trace(D_g * D_g');
%
%         mu_g = 1e2;
%         W_g = (Theta_g + mu_g * D_g) * (eye(L) + mu_g^2 * (D_g' * D_g))^(-0.5);
%
%         % Undershoot: double mu_g until overshoot
%         f_trial = fun_g(W_g);
%         while (f_trial - O_g) >= (mu_g * grad_norm_term)
%             mu_g = mu_g * 2;
%             W_g = (Theta_g + mu_g * D_g) * (eye(L) + mu_g^2 * (D_g' * D_g))^(-0.5);
%             f_trial = fun_g(W_g);
%         end
%
%         % Overshoot: halve mu_g until Armijo condition satisfied
%         while (f_trial - O_g) < (0.5 * mu_g * grad_norm_term) && (mu_g >= 1e-4)
%             mu_g = mu_g * 0.5;
%             W_g = (Theta_g + mu_g * D_g) * (eye(L) + mu_g^2 * (D_g' * D_g))^(-0.5);
%             f_trial = fun_g(W_g);
%         end
%
%         Theta(S, S) = W_g;
%         H = H_base + Hb_g * Theta(S, S) * Hf_g;
%     end
% end
%
% function [Theta] = step_armijo_nongeodesic_optimized(H_d, H_f, H_b, rho, Theta, D, L, G)
%     H = channel_aggregate(H_d, H_f, H_b, Theta);
%
%     for g = 1:G
%         S = (g - 1) * L + 1 : g * L;
%
%         Theta_g = Theta(S, S);
%         D_g     = D(S, S);
%
%         Hb_g = H_b(:, S);
%         Hf_g = H_f(S, :);
%
%         H_base = H - Hb_g * Theta_g * Hf_g;
%
%         fun_g = @(theta_block) rho' * svd(H_base + Hb_g * theta_block * Hf_g, 'econ');
%         O_g = fun_g(Theta_g);
%         grad_norm_term = 0.5 * trace(D_g * D_g');
%
%         mu_g = 1e2;
% 		X_g = Theta_g + mu_g * D_g;
%         W_g = X_g * (X_g' * X_g) ^ (-0.5);
%
%         % Undershoot: double mu_g until overshoot
%         f_trial = fun_g(W_g);
%         while (f_trial - O_g) >= (mu_g * grad_norm_term)
%             mu_g = mu_g * 2;
%             W_g = (Theta_g + mu_g * D_g) * (eye(L) + mu_g^2 * (D_g' * D_g))^(-0.5);
%             f_trial = fun_g(W_g);
%         end
%
%         % Overshoot: halve mu_g until Armijo condition satisfied
%         while (f_trial - O_g) < (0.5 * mu_g * grad_norm_term) && (mu_g >= 1e-4)
%             mu_g = mu_g * 0.5;
%             W_g = (Theta_g + mu_g * D_g) * (eye(L) + mu_g^2 * (D_g' * D_g))^(-0.5);
%             f_trial = fun_g(W_g);
%         end
%
%         Theta(S, S) = W_g;
%         H = H_base + Hb_g * Theta(S, S) * Hf_g;
%     end
% end
%
% function [G_r] = gradient_riemannian_nongeodesic(Theta, G_e)
% 	G_r = G_e - Theta * G_e' * Theta;
% end
%
% function [J, r, Theta] = scatter_singular_geodesic_modified(H_d, H_f, H_b, rho, L)
%     Theta = eye(size(H_f, 1));
%     H = channel_aggregate(H_d, H_f, H_b, Theta);
% 	[G_r, D] = deal(zeros(size(Theta)));
%     [iter.converge, iter.tolerance, iter.counter, iter.J] = deal(false, 1e-4, 0, rho' * svd(H, 'econ'));
%     while ~iter.converge
%         [iter.G_r, iter.D] = deal(G_r, D);
%         G_e = mask(gradient_euclidean(H, H_f, H_b, rho), L);
%         G_r = gradient_riemannian(Theta, G_e);
%         D = direction_conjugate(G_r, struct('G_r', iter.G_r, 'D', iter.D, 'counter', iter.counter));
%         Theta = step_armijo_geodesic_modified(H_d, H_f, H_b, rho, Theta, D, L);
%         H = channel_aggregate(H_d, H_f, H_b, Theta);
%         J = rho' * svd(H, 'econ');
%         iter.converge = (abs(J - iter.J) / abs(iter.J) <= iter.tolerance);
%         iter.J = J;
%         iter.counter = iter.counter + 1;
%     end
%     r = iter.counter;
% end
%
% function [Theta_new] = step_armijo_geodesic_modified(H_d, H_f, H_b, rho, Theta, D, L)
%     G = size(Theta, 1) / L;
%     Theta_new = Theta;
%
%     % Pre-calculate the full H matrix once.
%     H_full = channel_aggregate(H_d, H_f, H_b, Theta);
%
%     for g = 1:G
%         S = (g - 1) * L + 1 : g * L; % Indices for the current block
%
%         Theta_g = Theta(S, S);
%         D_g = D(S, S);
%
%         % Create a base H matrix by subtracting the current block's contribution.
%         H_contrib_old = H_b(:, S) * Theta_g * H_f(S, :);
%         H_base = H_full - H_contrib_old;
%
%         % Local cost function (the expensive part)
%         fun_g = @(theta_block) rho' * svd(H_base + H_b(:, S) * theta_block * H_f(S, :), 'econ');
%
%         % Cost at the current point
%         O_g = fun_g(Theta_g);
%
%         % --- OPTIMIZATION: Pre-calculate constant terms before the search ---
%         % 1. Calculate the trace term once, as it's constant for this block.
%         grad_norm_term = 0.5 * trace(D_g * D_g');
%
%         % 2. Pre-calculate the components of the matrix exponential.
%         [V, Dmat] = eig(D_g);
%         d_eig = diag(Dmat);
%
%         % --- Armijo search logic (STRUCTURE IS UNCHANGED) ---
%         mu_g = 10;
%
%         T_g = V * diag(exp(mu_g * d_eig)) * V';
%
%         % Undershoot loop (finding an upper bound)
%         while (fun_g(T_g^2 * Theta_g) - O_g) >= (mu_g * grad_norm_term)
%             mu_g = mu_g * 2;
%             T_g = V * diag(exp(mu_g * d_eig)) * V'; % Update T_g
%         end
%
%         % Overshoot loop (backtracking to meet condition)
%         armijo_check = 0.5 * mu_g * grad_norm_term;
%         while (fun_g(T_g * Theta_g) - O_g) < armijo_check && (mu_g >= eps)
%             mu_g = mu_g * 0.5;
%             armijo_check = armijo_check * 0.5; % Update check value
%             T_g = V * diag(exp(mu_g * d_eig)) * V'; % Update T_g
%         end
%
%         % Update the block and H_full for the next iteration
%         Theta_new(S, S) = T_g * Theta_g;
%         H_full = H_base + H_b(:, S) * Theta_new(S, S) * H_f(S, :);
%     end
% end
