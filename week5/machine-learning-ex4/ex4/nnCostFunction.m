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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% level 1
a_1 = [ones(m, 1), X];


% level 2
z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2, 1), 1), a_2];

% level 3
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);
h = a_3;

g = 0;
empty_y = zeros(num_labels, 1);

# y is a vector 
for _ = 1:m
    _y = empty_y;
    _y(y(_)) = 1;
    _h = h(_, :)';
    g = g + sum((-_y .* log(_h)) - ((1 - _y) .* log (1 - _h)));
end

t1 = Theta1;
t2 = Theta2;
t1(:, 1) = 0;
t2(:, 1) = 0;
regularizer = (lambda / (2*m)) * (sum(sum(t1.^2)) + sum(sum(t2.^2)));

J = ((1/m) * g) + regularizer;


% Backpropagation
for _ = 1:m
    
    % step 1
    i_a_1 = X (_, :)';
    i_a_1 = [1; i_a_1];
    
    i_z_2 = Theta1 * i_a_1;
    i_a_2 = sigmoid(i_z_2);
    i_a_2 = [1; i_a_2];
    
    i_z_3 = Theta2 * i_a_2;
    i_a_3 = sigmoid(i_z_3);

    % step 2
    %i_a_3 = a_3(_, :);
    %i_a_3 = i_a_3';
    _y = (1:num_labels == y(_))';
    d_3 = i_a_3 - _y;
    
    % step 3
    %i_z_2 = z_2(_, :);
    %i_z_2 = i_z_2';
    d_2 = (Theta2' * d_3)(2:end) .* sigmoidGradient(i_z_2);
    
    % step 4
    %i_a_1 = a_1(_, :);
    Theta1_grad = Theta1_grad + (d_2 * i_a_1');
    %i_a_2 = a_2(_, :);
    Theta2_grad = Theta2_grad + (d_3 * i_a_2');

end

% step 5
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% regularization

t1 = Theta1;
t1(:,1) = 0;
t2 = Theta2;
t2(:,1) = 0;
Theta1_grad = Theta1_grad + (lambda / m * t1);
Theta2_grad = Theta2_grad + (lambda / m * t2);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
