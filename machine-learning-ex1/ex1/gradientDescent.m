function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % X * alpha(2)

    %J_curr = theta * 
    col = [ 0 ; 1]
    partial_derivative = X * theta - y;
    tmp0 = theta(1) - alpha * sum(partial_derivative) / m;
    tmp1 = theta(2) - alpha * ((X(:, 2:2))' * partial_derivative) / m;
    theta(1) = tmp0;
    theta(2) = tmp1;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
