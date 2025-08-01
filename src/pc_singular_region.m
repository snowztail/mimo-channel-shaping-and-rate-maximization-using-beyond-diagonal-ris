clc; clear; close; setup;

[transmit.antenna, reflect.antenna, receive.antenna] = deal(3, 3, 3);
[channel.rank, reflect.bond] = deal(min(transmit.antenna, receive.antenna), reflect.antenna);
[channel.weight] = generate_weights(channel.rank, 1e2);
[number.weight] = size(channel.weight, 2);

channel.direct = zeros(receive.antenna, transmit.antenna);
channel.forward = fading_nlos(reflect.antenna, transmit.antenna);
channel.backward = fading_nlos(receive.antenna, reflect.antenna);

for w = 1 : number.weight
	clear scatter_power_max;
	[channel.sv(:, w)] = scatter_singular_geodesic(channel.direct, channel.forward, channel.backward, channel.weight(:, w), reflect.bond);
end

save('data/pc_singular_region.mat');



%% Plot function is vibe coded by Gemini ^_^
%% 1. Plot the Achievable Result
figure('Name', 'Simulated Singular Value Region vs Theoretical Outer Bounds', 'Color', 'white', 'Position', [0, 0, 500, 400]); % Use a white background
P = channel.sv';

% Create an alpha shape to capture the non-convex boundary.
% The 'alpha' parameter controls how tightly the shape wraps the points.
shp = alphaShape(P(:,1), P(:,2), P(:,3));

% Find the largest alpha value that produces a single, solid region.
% This automatically finds the best fit that introduces concavities
% (like the one needed for your Type 2 bound) without creating unwanted holes.
shp.Alpha = criticalAlpha(shp, 'one-region');

% Plot the resulting alpha shape
h_achievable = plot(shp);

% --- Set the visual properties to match your style ---
h_achievable.FaceColor = '#0072BD';
h_achievable.EdgeColor = '#002244';
h_achievable.LineWidth = 1;
h_achievable.FaceAlpha = 0.85;

material dull;
camlight head;
lighting gouraud;
hold on;


%% 2. Plot the Theoretical Bounds as Wireframes
f = svd(channel.forward, 'econ');
b = svd(channel.backward, 'econ');

boundaries = {
    @(x,y,z) x - f(1) * b(1), ...
    @(x,y,z) y - f(1) * b(2), ...
    @(x,y,z) y - f(2) * b(1), ...
    @(x,y,z) z - f(1) * b(3), ...
    @(x,y,z) z - f(2) * b(2), ...
    @(x,y,z) z - f(3) * b(1), ...
    @(x,y,z) x .* y - f(1) * f(2) * b(1) * b(2), ...
    @(x,y,z) x .* z - f(1) * f(3) * b(1) * b(2), ...
    @(x,y,z) y .* z - f(2) * f(3) * b(1) * b(2), ...
    @(x,y,z) x .* z - f(1) * f(2) * b(1) * b(3), ...
    @(x,y,z) y .* z - f(1) * f(2) * b(2) * b(3), ...
    @(x,y,z) y .* z - f(1) * f(3) * b(1) * b(3), ...
	@(x,y,z) x - y, ...
	@(x,y,z) y - z
};
boundaryTypes = [1,1,1,1,1,1, 2,2,2,2,2,2, 3,3];

% Use a professional color palette
typeColors = [
    0.8500 0.3250 0.0980; % Burnt Orange
    0.4660 0.6740 0.1880; % Leafy Green
    0.4940 0.1840 0.5560  % Deep Purple
];

% Define different line styles for each type
lineStyles = {'-', '--', ':'};

% Pre-allocate for the legend
numTypes = size(typeColors, 1);
legendHandles = gobjects(numTypes, 1);

% Use a dynamic plot range based on the data
min_vals = min(P,[],1); max_vals = max(P,[],1);
margin = (max_vals - min_vals) * 0.20; % 20% margin
plotRange = [min_vals(1)-margin(1), max_vals(1)+margin(1), ...
             min_vals(2)-margin(2), max_vals(2)+margin(2), ...
             min_vals(3)-margin(3), max_vals(3)+margin(3)];

for i = 1:numel(boundaries)
    func = boundaries{i};
    type = boundaryTypes(i);
    color = typeColors(type, :);
    style = lineStyles{type};

    h = fimplicit3(func, plotRange, ...
        'FaceColor', 'none', ...
        'EdgeColor', color, ...
        'LineStyle', style, ...
        'LineWidth', 1, ...
        'MeshDensity', 10); 

    if ~isgraphics(legendHandles(type))
        legendHandles(type) = h;
    end
end

%% 3. Add 2D Projections of the Achievable Result
% --- Define projection style ---
proj_fill_color = [0.4 0.4 0.4]; % A neutral dark gray for the fill
proj_alpha = 0.5;                % Set the transparency for the fill

% Get the actual boundary faces and vertices from the 3D alpha shape
[facets, vertices] = boundaryFacets(shp);

% --- Project onto XY plane (at z_min) ---
% Create the first patch and save its handle for the legend.
proj_vertices_xy = [vertices(:,1), vertices(:,2), ones(size(vertices,1), 1) * ax_limits(5)];
h_projection = patch('Faces', facets, 'Vertices', proj_vertices_xy, ...
      'FaceColor', proj_fill_color, ...
      'FaceAlpha', proj_alpha, ...      
      'EdgeColor', 'none');           % Using 'none' for a clean filled look

% --- Group the other projections to keep the legend clean ---
handle_group = hggroup('HandleVisibility', 'off'); 

% --- Project onto XZ plane (at y_max) ---
proj_vertices_xz = [vertices(:,1), ones(size(vertices,1), 1) * ax_limits(4), vertices(:,3)];
patch('Faces', facets, 'Vertices', proj_vertices_xz, ...
      'FaceColor', proj_fill_color, ...
      'FaceAlpha', proj_alpha, ...
      'EdgeColor', 'none', ...
      'Parent', handle_group); % Add to the invisible group

% --- Project onto YZ plane (at x_min) ---
proj_vertices_yz = [ones(size(vertices,1), 1) * ax_limits(1), vertices(:,2), vertices(:,3)];
patch('Faces', facets, 'Vertices', proj_vertices_yz, ...
      'FaceColor', proj_fill_color, ...
      'FaceAlpha', proj_alpha, ...
      'EdgeColor', 'none', ...
      'Parent', handle_group); % Add to the invisible group

%% 4. Finalize Plot
% Add the achievable region, bounds, and the new projection handle to the legend
finalHandles = [h_achievable; h_projection; legendHandles];
finalNames = {'Achieveable', '2D projection', 'Bounds (individual)', 'Bounds (product)', 'Bounds (ordering)'};

% Create the legend with the new entry
legend(finalHandles, finalNames, 'Location', 'northeast');

xlabel('$\sigma_1(\mathbf{H})$', 'Interpreter', 'latex'); 
ylabel('$\sigma_2(\mathbf{H})$', 'Interpreter', 'latex'); 
zlabel('$\sigma_3(\mathbf{H})$', 'Interpreter', 'latex');
hold off;
grid on;
axis equal;

savefig('plots/pc_singular_region.fig');



function [weights] = generate_weights(k, n)
	weights = find_integer_combos(k, n) * 1e6;
end

function [combos] = find_integer_combos(k, n)
	if k == 1
		combos = n;
		return;
	end
	
	combos = [];
	for i = 0 : n
		tail_combos = find_integer_combos(k - 1, n - i);
		top_row = repmat(i, 1, size(tail_combos, 2));
		new_combos = [top_row; tail_combos];
		combos = [combos, new_combos];
	end
end

function [G_e] = gradient_euclidean(H, H_f, H_b, rho)
	[U, ~, V] = svd(H, 'econ');
	G_e = H_b' * U * diag(rho) * V' * H_f';
end

function [sigma, Theta] = scatter_singular_geodesic(H_d, H_f, H_b, rho, L)
	N = size(H_f, 1);
	% Theta = eye(N);
	Theta = scatter_power_max(H_d, H_f, H_b, L);
	G = N / L;
	H = channel_aggregate(H_d, H_f, H_b, Theta);
	[G_e, G_r, D] = deal(zeros(size(Theta)));
	[iter.converge, iter.tolerance, iter.counter, iter.J] = deal(false, 1e-4, 0, rho' * svd(H));
	while ~iter.converge
		[iter.G_r, iter.D] = deal(G_r, D);
		for g = 1 : G
			S = (g - 1) * L + 1 : g * L;
			S_c = [1:S(1)-1, S(end)+1:N];
			fun = @(Theta_g) rho' * svd(H_d + H_b(:, S) * Theta_g * H_f(S, :) + H_b(:, S_c) * Theta(S_c, S_c) * H_f(S_c, :));
			G_e(S, S) = gradient_euclidean(H, H_f(S, :), H_b(:, S), rho);
			G_r(S, S) = gradient_riemannian(Theta(S, S), G_e(S, S));
			D(S, S) = direction_conjugate(G_r(S, S), struct('G_r', iter.G_r(S, S), 'D', iter.D(S, S), 'counter', iter.counter));
			Theta(S, S) = step_armijo_geodesic(fun, Theta(S, S), D(S, S));
			H = channel_aggregate(H_d, H_f, H_b, Theta);
		end
		sigma = svd(H, 'econ');
		J = rho' * sigma;
		iter.converge = (abs(J - iter.J) / iter.J <= iter.tolerance);
		iter.J = J;
		iter.counter = iter.counter + 1;
	end
end

function [Theta] = step_armijo_geodesic(fun, Theta, D)
	O = fun(Theta);
	mu = 1e4;
	T = expm(mu * D);
	% * Undershoot, double the step size
	while (fun(T ^ 2 * Theta) - O) >= (mu * 0.5 * trace(D * D'))
		mu = mu * 2;
		T = T ^ 2;
	end
	% * Overshoot, halve the step size
	while (fun(T * Theta) - O) < (0.5 * mu * 0.5 * trace(D * D')) && (mu >= eps)
		mu = mu * 0.5;
		T = expm(mu * D);
	end
	Theta = T * Theta;
end
