function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
digit = 0;
y_tmp = zeros(num_labels, 1);
X_tmp = zeros(1,size(X,2)+1);
J_tmp = zeros(m, 1);

for i = 1:m,

    X_tmp = [1 X(i,:)]; % get one example from 5000 examples
    digit = y(i); % get the digit of the example
    
    y_tmp(digit,1) = 1; % initialize the example's result vector
    a2 = [1; sigmoid(Theta1 * X_tmp')]; % compute the hidden layer's activation
    a3 = sigmoid(Theta2 * a2); % compute the output

    J_tmp(i,1) = J_tmp(i,1) - y_tmp' * log(a3) - (1 - y_tmp)' * log(1-a3);
   % for k = 1:num_labels, % for each label, compute the example's cost 
        %y_tmp(k,1) = k;
       % J_tmp(i,1) = J_tmp(i,1) - y_tmp(k,1) * log(a3(k,1)) - (1 - y_tmp(k,1)) * log(1-a3(k,1));
    %end
    
    J = J + J_tmp(i,1);  
    y_tmp(digit,1) = 0; % set y_tmp elements back to zeros
end
J = J/m;

theta1_reg = 0;
theta2_reg = 0;
for j = 1:size(Theta1,1),
    for k = 2:size(Theta1,2),
        theta1_reg = theta1_reg + Theta1(j,k)^2;
    end
end
for j = 1:size(Theta2,1),
    for k = 2:size(Theta2,2),
        theta2_reg = theta2_reg + Theta2(j,k)^2;
    end
end

J = J + (theta1_reg + theta2_reg) * lambda/(2*m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
a1 = zeros(1,size(X,2)+1);
Delta1 = zeros(hidden_layer_size, input_layer_size+1);
Delta2 = zeros(num_labels, hidden_layer_size+1);
for i = 1:m,
    X_tmp = [1 X(i,:)]; % get one example from 5000 examples
    a1 = X_tmp;
    
    digit = y(i); % get the digit of the example
    y_tmp(digit,1) = 1; % initialize the example's result vector
    
    a2 = [1; sigmoid(Theta1 * a1')]; % compute the hidden layer's activation
    a3 = sigmoid(Theta2 * a2); % compute the output
    
    % for each output unit k in layer 3
    delta3 = zeros(num_labels,1);
    for k = 1:num_labels,
        delta3(k,1) = a3(k,1) - y_tmp(k,1);
    end
    
    % for the hidden layer
    delta2 = (Theta2(:, 2:end))' * delta3 .* ...
        sigmoidGradient(Theta1 * X_tmp');
    % accumulate the gradient from this example
    
    Delta2 = Delta2 + delta3 * (a2)';
    %size(a1)
    %size(delta2(2:end,1))
    Delta1 = Delta1 + delta2 * a1;
        
    y_tmp(digit,1) = 0; % set y_tmp elements back to zeros
end
% obtain the (unregularized) gradient for the NN cost
% function
Theta2_grad = Delta2/m;
Theta1_grad = Delta1/m;
% regularize NN
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ...
    Theta2(:,2:end) * lambda/m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ...
    Theta1(:,2:end) * lambda/m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
