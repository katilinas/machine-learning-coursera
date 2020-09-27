function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
  fprintf('iter ... \n');
  fprintf('%f\n', iter);
  
  summa = sum((X*theta - y).*X)
  fprintf('summa ... \n');
  fprintf('%f\n', summa);
  
  delta = (alpha/length(y))*(summa)
  fprintf('delta ... \n');
  fprintf('%f\n', delta);
  fprintf('delta size ... \n');
  fprintf('%f\n', size(delta));
  
  fprintf('theta before... \n');
  fprintf('%f\n', theta);
  fprintf('theta before size ... \n');
  fprintf('%f\n', size(theta));
  
	theta = theta - delta';
	fprintf('theta after... \n');
  fprintf('%f\n', theta);
  fprintf('theta after size ... \n');
  fprintf('%f\n', size(theta));
  
  costs = computeCost(X, y, theta);
  fprintf('costs ... \n');
  fprintf('%f\n', costs);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = costs;

end

end
