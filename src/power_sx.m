clc; clear; close; setup;

[transmit.antenna, reflect.antenna, receive.antenna] = deal(16, 2 .^ (4 : 8), 16);
% [channel.pathloss.distance.direct, channel.pathloss.distance.forward, channel.pathloss.distance.backward, channel.pathloss.exponent.direct, channel.pathloss.exponent.forward, channel.pathloss.exponent.backward] = deal(-14.7, -10, -6.3, -3, -2.4, -2);
[channel.pathloss.direct, channel.pathloss.forward, channel.pathloss.backward] = deal(db2pow(-65), db2pow(-54), db2pow(-46));
[number.bond, number.antenna, number.realization, flag.direct] = deal(log2(reflect.antenna) + 1, length(reflect.antenna), 1e2, false);

for r = 1 : number.realization
	% * No RIS
	channel.direct = flag.direct * sqrt(channel.pathloss.direct) * fading_nlos(receive.antenna, transmit.antenna);
	channel.power.direct(r) = norm(channel.direct, 'fro') ^ 2;
	% * Have RIS
	for a = 1 : number.antenna
		channel.forward = sqrt(channel.pathloss.forward) * fading_nlos(reflect.antenna(a), transmit.antenna);
		channel.backward = sqrt(channel.pathloss.backward) * fading_nlos(receive.antenna, reflect.antenna(a));
		clear scatter_power_max;
		for b = 1 : number.bond(a)
			reflect.bond = 2 .^ (b - 1);
			[reflect.beamformer, channel.aggregate] = scatter_power_max(channel.direct, channel.forward, channel.backward, reflect.bond);
			channel.power.aggregate(b, a, r) = norm(channel.aggregate, 'fro') ^ 2;
		end
		if flag.direct
			[~, channel.procrustes.left] = scatter_procrustes_left(channel.direct, channel.forward, channel.backward);
			[~, channel.procrustes.right] = scatter_procrustes_right(channel.direct, channel.forward, channel.backward);
			channel.power.procrustes.left(a, r) = norm(channel.procrustes.left, 'fro') ^ 2;
			channel.power.procrustes.right(a, r) = norm(channel.procrustes.right, 'fro') ^ 2;
		end
	end
end
channel.power.direct = mean(channel.power.direct, ndims(channel.power.direct));
channel.power.aggregate = mean(channel.power.aggregate, ndims(channel.power.aggregate));
if flag.direct
    channel.power.procrustes.left = mean(channel.power.procrustes.left, ndims(channel.power.procrustes.left));
    channel.power.procrustes.right = mean(channel.power.procrustes.right, ndims(channel.power.procrustes.right));
end
% save('data/power_sx.mat');

figure('Name', 'Channel Power vs RIS Configuration', 'Position', [0, 0, 500, 400]);
if flag.direct
    handle.power.direct = semilogy(1 : max(number.bond), repmat(channel.power.direct, [1, max(number.bond)]), 'Color', 'k', 'Marker', 'none', 'DisplayName', 'No RIS');
    hold on;
	for a = 1 : number.antenna
		handle.power.procrustes.left(a) = scatter(number.bond(a), channel.power.procrustes.left(a), 'Marker', '<');
		hold on;
		handle.power.procrustes.right(a) = scatter(number.bond(a), channel.power.procrustes.right(a), 'Marker', '>');
		hold on;
	end
	handle.power.procrustes.dummy.left = scatter(nan, nan, 'MarkerEdgeColor', '#808080', 'Marker', '<', 'DisplayName', 'OP-left');
	hold on;
	handle.power.procrustes.dummy.right = scatter(nan, nan, 'MarkerEdgeColor', '#808080', 'Marker', '>', 'DisplayName', 'OP-right');
	hold on;
	set(handle.power.procrustes.left, {'MarkerEdgeColor'}, {'#0072BD'; '#D95319'; '#EDB120'; '#7E2F8E'; '#77AC30'});
	set(handle.power.procrustes.right, {'MarkerEdgeColor'}, {'#0072BD'; '#D95319'; '#EDB120'; '#7E2F8E'; '#77AC30'});
end
for a = 1 : number.antenna
	handle.power.aggregate(a) = semilogy(1 : number.bond(a), channel.power.aggregate(1 : number.bond(a), a), 'DisplayName', '$N_\mathrm{S} = ' + string(reflect.antenna(a)) + '$');
    hold on;
end
style_plot(handle.power.aggregate);
set(handle.power.aggregate, {'Marker'}, {'none'});
if flag.direct
	legend([handle.power.direct, handle.power.procrustes.dummy.left, handle.power.procrustes.dummy.right, handle.power.aggregate], 'Location', 'nw');
else
	legend('Location', 'nw');
end
hold off; grid on; box on;
set(gca, 'XLim', [1, max(number.bond)], 'XTick', 1 : max(number.bond), 'XTickLabel', '$2^' + string(vec(0 : max(number.bond) - 1)) + '$', 'YLim', [channel.power.direct * 0.95, max(vec(channel.power.aggregate)) * 1.05]);
xlabel('RIS Group Size');
ylabel('Channel Power [W]');
% savefig('plots/power_sx.fig');
% matlab2tikz('../assets/simulation/power_sx.tex', 'width', '8cm', 'height', '6cm');
