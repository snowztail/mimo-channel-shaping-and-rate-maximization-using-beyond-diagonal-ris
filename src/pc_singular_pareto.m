clc; clear; close; setup;

[transmit.antenna, reflect.antenna, receive.antenna] = deal(2, 16, 2);
[channel.rank, reflect.bond] = deal(min(transmit.antenna, receive.antenna), unique([1, 4, 16, reflect.antenna]));
assert(channel.rank == 2, 'The plot function is only for 2-sv case.');
[channel.pathloss.direct, channel.pathloss.forward, channel.pathloss.backward] = deal(db2pow(-65), db2pow(-54), db2pow(-46));
[channel.weight(1, :, 1), channel.weight(2, :, 1)] = pol2cart(pi / 4 : pi / 64 : pi * 3 / 4, 1);
[channel.weight(1, :, 2), channel.weight(2, :, 2)] = pol2cart(pi / 4 : -pi / 64 : -pi / 4, 1);
[channel.weight(1, :, 3), channel.weight(2, :, 3)] = pol2cart(5 * pi / 4 : pi / 64 : pi * 7 / 4, 1);
[channel.weight(1, :, 4), channel.weight(2, :, 4)] = pol2cart(5 * pi / 4 : -pi / 64 : pi * 3 / 4, 1);
[number.weight, number.case, number.bond, number.realization, flag.direct] = deal(size(channel.weight, 2), size(channel.weight, 3), length(reflect.bond), 2, false);

for r = 1 : number.realization
	channel.direct = flag.direct * sqrt(channel.pathloss.direct) * fading_nlos(receive.antenna, transmit.antenna);
	channel.forward = sqrt(channel.pathloss.forward) * fading_nlos(reflect.antenna, transmit.antenna);
	channel.backward = sqrt(channel.pathloss.backward) * fading_nlos(receive.antenna, reflect.antenna);
	channel.singular.direct(:, r) = svd(channel.direct);
	for b = 1 : number.bond
		for c = 1 : number.case
			for w = 1 : number.weight
				clear scatter_singular;
				[reflect.beamformer, channel.aggregate] = scatter_singular(channel.direct, channel.forward, channel.backward, channel.weight(:, w, c), reflect.bond(b));
				channel.singular.aggregate(:, w, c, b, r) = svd(channel.aggregate);
			end
		end
	end
end
channel.singular.direct = mean(channel.singular.direct, ndims(channel.singular.direct));
channel.singular.aggregate = mean(channel.singular.aggregate, ndims(channel.singular.aggregate));
save('data/pc_singular_pareto.mat');

figure('Name', 'Channel Singular Value Pareto Front vs RIS Group Size', 'Position', [0, 0, 500, 400]);
handle.color = {'#0072BD'; '#D95319'; '#EDB120'; '#7E2F8E'; '#77AC30'; '#4DBEEE'; '#A2142F'};
hold all;
handle.singular.direct = scatter(channel.singular.direct(1), channel.singular.direct(2), 200, 'k', '^', 'DisplayName', 'Direct');
for b = 1 : number.bond
	point.all = [vec(channel.singular.aggregate(1, :, :, b)), vec(channel.singular.aggregate(2, :, :, b))];
	% * Pareto frontier
	point.pareto = point.all(convhull(point.all), :);
	handle.singular.pareto(b) = plot(point.pareto(:, 1), point.pareto(:, 2), 'DisplayName', '$L = ' + string(reflect.bond(b)) + '$', 'LineStyle', ':', 'Color', handle.color{b});
	% * channel power gain-optimal point
	[~, index.power] = max(vecnorm(point.pareto, 2, 2));
	if ~flag.direct && b == number.bond
		% ! from analysis we know
		index.power = find(vecnorm(point.pareto, 2, 2) >= (1 - 5e-2) * max(vecnorm(point.pareto, 2, 2)));
	end
	point.power = point.pareto(index.power, :);
	handle.singular.power(b) = scatter(point.power(:, 1), point.power(:, 2), 200, 'MarkerEdgeColor', handle.color{b});
	% * power transfer-optimal point
	[~, index.energy] = max(point.pareto(:, 1));
	point.energy = point.pareto(index.energy, :);
	handle.singular.energy(b) = scatter(point.energy(1), point.energy(2), 200, 'Marker', 'x', 'MarkerEdgeColor', handle.color{b});
	% * rate-optimal arc
	[~, index.rate.low] = max(point.pareto(:, 1));
	[~, index.rate.high] = max(point.pareto(:, 2));
	index.rate.range = circshift(setdiff(1 : size(point.pareto, 1), min(index.rate.low, index.rate.high) + 1 : max(index.rate.low, index.rate.high) - 1), -index.rate.high);
	handle.singular.rate(b) = plot(point.pareto(index.rate.range, 1), point.pareto(index.rate.range, 2), 'LineStyle', '-', 'Color', handle.color{b});
end
handle.dummy.power = scatter(nan, nan, 200, 'MarkerEdgeColor', '#808080', 'DisplayName', 'P-max');
handle.dummy.energy = scatter(nan, nan, 200, 'Marker', 'x', 'MarkerEdgeColor', '#808080', 'DisplayName', 'E-max');
handle.dummy.rate = plot(nan, nan, 'Color', '#808080', 'DisplayName', 'R-max');
hold off; grid on; xlim tight; ylim tight; box on; legend([handle.singular.direct, handle.singular.pareto, handle.dummy.power, handle.dummy.energy, handle.dummy.rate], 'Location', 'nw');
xlabel('$\sigma_1(\mathbf{H})$');
ylabel('$\sigma_2(\mathbf{H})$');
savefig('plots/pc_singular_pareto.fig');
matlab2tikz('../assets/simulation/pc_singular_pareto.tex', 'width', '8cm', 'height', '6cm', 'extraaxisoptions', {'every axis plot/.append style={line width=1.5pt}'});
