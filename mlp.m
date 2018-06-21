% error metric
MSE = @(e, N) = e'*e/N;

A = neuron(x, W);
y = sum(A);
e = (d - y).^2;

% forward evalutation
y = sum([neuron(W(1,:), x) neuron(W(2,:), x]);

% backpropagation
