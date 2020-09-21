function [J, grad] = costFunction(X, y, theta, lambda)
% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h=X*theta;
t=theta;
t(1)=0;
J=1/(2*m)*sum((h-y).^2)+lambda/(2*m)*sum(t.^2);
grad = 1/m * X'*(h-y)+ lambda/m*t;
grad = grad(:);

end
