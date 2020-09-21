function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters,lambda)
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
h=X*theta;

temp=h-y;

temp0=theta(1)-alpha*(1/m)*sum(temp);


temp1=theta(2)-alpha*(1/m)*sum(temp.*X(:,2));

theta=[temp0;temp1];

J_history(iter) = costFunction(X, y, theta,lambda);

end

end
