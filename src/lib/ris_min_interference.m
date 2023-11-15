function [Theta] = ris_min_interference(H_d, H_f, H_b, Theta, G)
	[K, L] = deal(size(H_d, 3), length(Theta) / G);
	[iter.converge, iter.tolerance, iter.counter] = deal(false, 1e-4, 0);
	H = channel_aggregate(H_d, H_f, H_b, Theta);
	I = interference_leakage(H)
	while ~iter.converge
		iter.I = I;
		for g = 1 : G
			S = (g - 1) * L + 1 : g * L;
			S_c = setdiff(1 : length(Theta), S);
			D = H_d + pagemtimes(pagemtimes(H_b(:, S_c, :), Theta(S_c, S_c)), H_f(S_c, :, :, :));
            T = pagemtimes(H_b(:, S, :), 'ctranspose', H_b(:, S, :), 'none');
			B = 2 * max(pageeig(T)) .* repmat(eye(L), [1, 1, K]) - T;
			Q = pagemtimes(pagemtimes(pagemtimes(B, Theta(S, S)), H_f(S, :, :, :)) - pagemtimes(pagectranspose(H_b(:, S, :)), D), pagectranspose(H_f(S, :, :, :)));
			M = sum(Q(:, :, ~logical(eye(K))), [3, 4]);
			[U, ~, V] = svd(M);
			Theta(S, S) = U * V';
		end
		H = channel_aggregate(H_d, H_f, H_b, Theta);
		I = interference_leakage(H)
		iter.converge = (abs(I - iter.I) <= iter.tolerance);
		iter.counter = iter.counter + 1;
	end
end
