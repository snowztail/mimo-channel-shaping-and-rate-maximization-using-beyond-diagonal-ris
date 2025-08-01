clc; clear; close; setup;

[transmit.antenna, reflect.antenna, receive.antenna] = deal(4, [16, 64, 256], 4);
[transmit.power, receive.noise] = deal(db2pow(20), db2pow(-75));
[channel.pathloss.direct, channel.pathloss.forward, channel.pathloss.backward] = deal(db2pow(-65), db2pow(-54), db2pow(-46));
[number.antenna, number.realization, flag.direct] = deal(length(reflect.antenna), 1e3, true);

for r = 1 : number.realization
	channel.direct = flag.direct * sqrt(channel.pathloss.direct) * fading_nlos(receive.antenna, transmit.antenna);
	for a = 1 : number.antenna
		channel.forward = sqrt(channel.pathloss.forward) * fading_nlos(reflect.antenna(a), transmit.antenna);
		channel.backward = sqrt(channel.pathloss.backward) * fading_nlos(receive.antenna, reflect.antenna(a));

		clear scatter_rate_optimized;
		transmit.beamformer.diagonal = precoder_rate(channel.direct, transmit.power, receive.noise);
		[iter.converge, iter.tolerance, iter.counter, iter.rate] = deal(false, 1e-4, 0, rate_mimo(channel.direct, transmit.beamformer.diagonal, receive.noise));
		tic;
		while ~iter.converge && iter.counter < 1e2
			[reflect.beamformer.diagonal, channel.aggregate.diagonal] = scatter_rate_optimized(channel.direct, channel.forward, channel.backward, transmit.beamformer.diagonal, 1, receive.noise);
			transmit.beamformer.diagonal = precoder_rate(channel.aggregate.diagonal, transmit.power, receive.noise);
			receive.rate.diagonal(a, r) = rate_mimo(channel.aggregate.diagonal, transmit.beamformer.diagonal, receive.noise);
			iter.converge = (abs(receive.rate.diagonal(a, r) - iter.rate) / iter.rate <= iter.tolerance);
			iter.rate = receive.rate.diagonal(a, r);
			iter.counter = iter.counter + 1;
		end
		[objective.diagonal(a, r), counter.diagonal(a, r), timer.diagonal(a, r)] = deal(receive.rate.diagonal(a, r), iter.counter, toc);


		clear scatter_rate_optimized;
		transmit.beamformer.unitary = precoder_rate(channel.direct, transmit.power, receive.noise);
		[iter.converge, iter.tolerance, iter.counter, iter.rate] = deal(false, 1e-4, 0, rate_mimo(channel.direct, transmit.beamformer.unitary, receive.noise));
		tic;
		while ~iter.converge && iter.counter < 1e2
			[reflect.beamformer.unitary, channel.aggregate.unitary] = scatter_rate_optimized(channel.direct, channel.forward, channel.backward, transmit.beamformer.unitary, reflect.antenna(a), receive.noise);
			transmit.beamformer.unitary = precoder_rate(channel.aggregate.unitary, transmit.power, receive.noise);
			receive.rate.unitary(a, r) = rate_mimo(channel.aggregate.unitary, transmit.beamformer.unitary, receive.noise);
			iter.converge = (abs(receive.rate.unitary(a, r) - iter.rate) / iter.rate <= iter.tolerance);
			iter.rate = receive.rate.unitary(a, r);
			iter.counter = iter.counter + 1;
		end
		[objective.unitary(a, r), counter.unitary(a, r), timer.unitary(a, r)] = deal(receive.rate.unitary(a, r), iter.counter, toc);
	end
end

[objective.diagonal, objective.unitary] = deal(mean(objective.diagonal, ndims(objective.diagonal)), mean(objective.unitary, ndims(objective.unitary)));
[counter.diagonal, counter.unitary] = deal(mean(counter.diagonal, ndims(counter.diagonal)), mean(counter.unitary, ndims(counter.unitary)));
[timer.diagonal, timer.unitary] = deal(mean(timer.diagonal, ndims(timer.diagonal)), mean(timer.unitary, ndims(timer.unitary)));

save('data/pc_complexity_bond.mat');

fprintf('Objective: %.4g, %.4g\n', [objective.diagonal, objective.unitary]');
fprintf('Counter:   %.4g, %.4g\n', [counter.diagonal, counter.unitary]');
fprintf('Timer:     %.4g, %.4g\n', [timer.diagonal, timer.unitary]');



function [G_e] = gradient_euclidean(H, H_f, H_b, W, P_n)
	G_e = H_b' * H * W / (eye(size(W, 2)) + (H * W)' * (H * W) / P_n) * W' * H_f' / P_n;
end

function [Theta, H] = scatter_rate_optimized(H_d, H_f, H_b, W, L, P_n)
    % Initialize the reflecting beamformer as an identity matrix
    N = size(H_f, 1);
    Theta = eye(N);

    % Pre-calculate constants and initial state
    G = N / L; % Number of blocks
    H = channel_aggregate(H_d, H_f, H_b, Theta);

    % Initialize gradients and search direction matrices
    [G_r, D] = deal(zeros(N));

    % Set up iteration parameters
    [iter.converge, iter.tolerance, iter.counter, iter.cycle] = deal(false, 1e-4, 0, N^2);
    iter.R = rate_mimo(H, W, P_n); % Initial rate

    % Main optimization loop
    while ~iter.converge && iter.counter < 50 % Add a max iteration guard
        % Store previous iteration's gradient and direction
        [iter.G_r, iter.D] = deal(G_r, D);

        % 1. Calculate Euclidean gradient for the entire matrix
        G_e = mask(gradient_euclidean(H, H_f, H_b, W, P_n), L);

        % 2. Project to Riemannian gradient on the manifold
        G_r = gradient_riemannian(Theta, G_e); % Assumes gradient_riemannian is available

        % 3. Determine conjugate search direction
        D = direction_conjugate_optimized(G_r, iter.G_r, iter.D, iter.counter, iter.cycle);

        % 4. Perform efficient line search to update Theta
        Theta = step_armijo_geodesic_optimized(H_d, H_f, H_b, W, P_n, Theta, D, L, G);

        % 5. Update channel and check for convergence
        H = channel_aggregate(H_d, H_f, H_b, Theta);
        R = rate_mimo(H, W, P_n);
        iter.converge = (abs(R - iter.R) / iter.R <= iter.tolerance);

        % Update for next iteration
        iter.R = R;
        iter.counter = iter.counter + 1;
    end
end

function [Theta] = step_armijo_geodesic_optimized(H_d, H_f, H_b, W, P_n, Theta, D, L, G)
    % Calculate the full aggregate channel matrix once at the start.
    H = channel_aggregate(H_d, H_f, H_b, Theta);

    % Sequentially update each block of the Theta matrix
    for g = 1:G
        S = (g - 1) * L + 1 : g * L; % Indices for the current block

        Theta_g = Theta(S, S);
        D_g     = D(S, S);

        % Isolate the parts of the channel matrices related to the current block
        Hb_g = H_b(:, S);
        Hf_g = H_f(S, :);

        % Calculate the base channel, excluding the current block's contribution.
        % This is the key optimization, as H_base is constant during the line search for this block.
        H_base = H - Hb_g * Theta_g * Hf_g;

        % *** THIS IS THE CRITICAL CHANGE ***
        % The local cost function is now rate_mimo, not svd.
        fun_g = @(theta_block) rate_mimo(H_base + Hb_g * theta_block * Hf_g, W, P_n);

        % Cost at the current point for this block
        O_g = fun_g(Theta_g);

        % Pre-calculate the gradient norm term for the Armijo condition
        grad_norm_term = 0.5 * trace(D_g * D_g');

        % Efficiently compute matrix exponential via eigendecomposition (once per block)
        [V, d_eig] = eig(D_g, 'vector');

        % --- Armijo Search Logic ---
        mu_g = 10; % Initial step size

        % Undershoot loop: find an upper bound for the step size
        T_g = V * diag(exp(mu_g * d_eig)) * V';
        while (fun_g(T_g^2 * Theta_g) - O_g) >= (mu_g * grad_norm_term) && mu_g < 1e4
            mu_g = mu_g * 2;
            T_g = V * diag(exp(mu_g * d_eig)) * V';
        end

        % Overshoot loop: backtrack to satisfy the Armijo condition
        armijo_check = 0.5 * mu_g * grad_norm_term;
        while (fun_g(T_g * Theta_g) - O_g) < armijo_check && (mu_g >= 1e-4)
            mu_g = mu_g * 0.5;
            armijo_check = armijo_check * 0.5;
            T_g = V * diag(exp(mu_g * d_eig)) * V';
        end

        % Update the current block of Theta
        Theta(S, S) = T_g * Theta_g;

        % Update the full channel matrix with the new block's contribution
        % This ensures the next block's optimization sees the most recent update.
        H = H_base + Hb_g * Theta(S, S) * Hf_g;
    end
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

function [Theta] = mask(Theta, L)
	G = length(Theta) / L;
	Theta = Theta .* kron(eye(G), ones(L));
end
