function [Theta, H] = scatter_singular(H_d, H_f, H_b, rho, L)
	persistent iter;
	if isempty(iter)
		clear scatter_power_max scatter_power_min;
		if all(rho >= 0)
			Theta = scatter_power_max(H_d, H_f, H_b, L);
		elseif all(rho <= 0)
			Theta = scatter_power_min(H_d, H_f, H_b, L);
		else
			Theta = eye(size(H_f, 1));
		end
	else
		Theta = iter.Theta;
	end

	G = length(Theta) / L;
	H = channel_aggregate(H_d, H_f, H_b, Theta);
	[G_e, G_r, D] = deal(zeros(size(Theta)));
	[iter.converge, iter.tolerance, iter.counter, iter.J] = deal(false, 1e-9, 0, rho' * svd(H));
	while ~iter.converge
		[iter.G_r, iter.D] = deal(G_r, D);
		for g = 1 : G
			S = (g - 1) * L + 1 : g * L;
			S_c = setdiff(1 : length(Theta), S);
			fun = @(Theta_g) rho' * svd(H_d + H_b(:, S) * Theta_g * H_f(S, :) + H_b(:, S_c) * Theta(S_c, S_c) * H_f(S_c, :));
			G_e(S, S) = gradient_euclidean(H, H_f(S, :), H_b(:, S), rho);
			G_r(S, S) = gradient_riemannian(Theta(S, S), G_e(S, S));
			D(S, S) = direction_conjugate(G_r(S, S), struct('G_r', iter.G_r(S, S), 'D', iter.D(S, S), 'counter', iter.counter));
			Theta(S, S) = step_armijo(fun, Theta(S, S), D(S, S));
			H = channel_aggregate(H_d, H_f, H_b, Theta);
		end
		J = rho' * svd(H);
		iter.converge = (abs(J - iter.J) / iter.J <= iter.tolerance);
		iter.J = J;
		iter.counter = iter.counter + 1;
	end
	iter.Theta = Theta;
end

function [G_e] = gradient_euclidean(H, H_f, H_b, rho)
	[U, ~, V] = svd(H, 'econ');
	G_e = H_b' * U * diag(rho) * V' * H_f';
end
