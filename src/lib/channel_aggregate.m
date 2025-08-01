function [H] = channel_aggregate(H_d, H_f, H_b, Theta)
	% if size(H_d, 3) == 1
		H = H_d + H_b * Theta * H_f;
	% else
	% 	H = H_d + pagemtimes(pagemtimes(H_b, Theta), H_f);
	% end
end
