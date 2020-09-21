
%% Project1: Nanjing Housing Price Prediction
% Implement linear regression (Analytic, GD) for Nanjing housing 
% price prediction

%% Initialization
clear ; close all; clc

%% Part 1: Loading and Plotting
X = [2000, 2001, 2002, 2003, 2004, 2005, 2006, ...
    2007, 2008, 2009, 2010, 2011, 2012, 2013]';
y = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704,...
    6.853, 7.971, 8.561, 10.000, 11.280, 12.900]';

m = length(y); % number of training examples


% feature normalize
[X_norm,X_mu,X_sigma]=featureNormalize(X);
[y_norm,y_mu,y_sigma]=featureNormalize(y);

% plotting
plotData(X_norm, y_norm);

%% Part 2:Cost and Gradient descent
X_norm=[ones(m, 1), X_norm]; % Add a column of ones to x
% random initialization theta
theta = randInitializeTheta(2,1);
iterations = 3000;
alpha = 0.01;
lambda=1;
fprintf('\nRunning Gradient Descent ...\n');
% run gradient descent
[theta,J_history] = gradientDescent(X_norm, y_norm, theta, alpha, iterations,lambda);
% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

% Plot the linear fit
hold on; % keep previous plot visible
plot(X_norm(:,2), X_norm*theta, '-');
legend('Training data', 'Linear regression');
hold off; % don't overlay any more plots on this figure

% predict X=2014
Xt=2014;
Xt_norm=Xt-X_mu;
Xt_norm=Xt_norm/X_sigma;
Xt_norm=[1,Xt_norm];
yt_norm=Xt_norm*theta;
yt=reFeature(yt_norm,y_mu,y_sigma);

fprintf('\nPredict the Nanjing housing price in 2014.\n')
fprintf('For X = 2014, we predict price is  %f\n',yt);

%% Part 3: Normal equations
% Using normal equations to solve theta
theta_e=normalEqn(X_norm,y_norm);

% print theta to screen
fprintf('Theta found by close-form solution:\n');
fprintf('%f\n', theta_e);

% predict X=2014
Xt=2014;
Xt_norm=Xt-X_mu;
Xt_norm=Xt_norm/X_sigma;
Xt_norm=[1,Xt_norm];
yt_norm=Xt_norm*theta_e;
yt=reFeature(yt_norm,y_mu,y_sigma);

fprintf('\nPredict the Nanjing housing price in 2014.\n')
fprintf('For X = 2014, we predict price is %f\n',yt);

%% Part 4: Visualizing J(theta_0, theta_1)
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = costFunction(X_norm, y_norm, t,1);
    end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
