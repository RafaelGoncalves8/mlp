
%plot inline

!cat ../data/winequality.names

!head ../data/winequality-red.csv

data = dlmread("../data/winequality-red.csv", ";" ,1, 0); % skip feature names
size(data)

% randomize order of data for excluding biases in the dataset
n = rand(length(data),1);
[_ index] = sort(n);
data_rand = data(index, :);

data_norm = [];
mu = [];
sigma = [];

% normalizing data for optimum use of algorithms
for j = 1:size(data_rand,2),
    mu = [mu; mean(data_rand(:,j))];
    sigma = [sigma; std(data_rand(:,j))];
    data_norm = [data_norm, (data_rand(:,j)- mu(j)*ones(size(data,1),1))/sigma(j)];
end

% correlation matrix
cor = corr(data_norm)

imagesc(cor);

set(gca, 'XTick', 1:size(cor,2)); % center x-axis ticks on bins
set(gca, 'YTick', 1:size(cor,2)); % center y-axis ticks on bins
set(gca, 'YTickLabel', ["fixed acidity - 1";"volatile acidity - 2";"citric acid - 3";"residual sugar - 4";
                        "chlorides - 5";"free sulfur dioxide - 6";"total sulfur dioxide - 7";"density - 8";
                        "pH - 9";"sulphates - 10";"alcohol - 11";"quality - 12"]); % set y-axis labels
                        
title('Correlation Matrix', 'FontSize', 14); % set title

caxis([-1, 1]);  % colorbar ranging from -1 to 1
colormap('jet'); % set the colorscheme
colorbar;        % enable colorbar

X = data_norm(:,1:11); % inputs
y = data_norm(:, 12);  % labels

X_train = X(1:1119, :); % 70% for training
y_train = y(1:1119);

X_test = X(1120:1599, :);  % 30% for testing
y_test = y(1120:1599);

% M number of labeled inputs
% N number of features (lenght of input vector)
[M, N] = size(X_train)

O = 8 % number of neurons in the hidden layer

% initial weights matrix as small random values
W = rand([O N]).*0.01; % W: OxN

% adding column for bias
X_train_bias = [ones(size(X_train,1),1), X_train]'; % X_train_bias: MxN+1
X_test_bias = [ones(size(X_test,1),1), X_test]';
W_bias = [ones(size(W,1),1), W]; % W_bias: N+1xO

function Delta = get_delta(u, e),
    F_prime = zeros(size(u,2), size(u,2));
    for i = 1:size(u,2),
        for j = 1:size(u,2),
            if i==j,
                F_prime(i,j) = (sech(u(1)))^2; % F: OxO
            end
        end
    end
    
    Delta = -2*F_prime*e; % delta: OxO
    
end

function [W, MSE] = backprop_batch_step(X, s, W, alpha),

    [M, N] = size(X);
    
    aux = zeros(size(W));
    E = [];
    
    for i = 1:M,
        % feedfoward
        u = X(:,i)'*W';      % u: 1xO
        a = tanh(u);         % output of hidden layer a: 1xO
        y = sum(a);          % y: 1x1

        e = s(i) - y;
        
        E = [E; e]; % E: Mx1
        
        Delta = get_delta(u, e);           % Delta: OxO
        Xv = ones(size(Delta), 1)*X(:,i)'; % Xv: OxN
        aux = aux + Delta*Xv;              % aux: OxN
    end

    aux = aux/M;
                              
    W = W - alpha.*aux; 

    % metric
    MSE = (E'*E)/M;
end

function [W, mse_train_vec, mse_test_vec] = batch_backpropagation (X_te, X_tr, y_te, y_tr, W, alpha, epsilon, gamma,
                                                                    show_steps, max_iter),
    mse_test_vec = [];
    mse_train_vec = [];
    mse_test = epsilon+1;
    mse_min = inf;
    counter = 0;
    iter = 0;
    
    % stop if passed gamma steps without improvement 
    % or if error is less than epsilon
    % or if it is the iteration number max_iter
    while counter < gamma && mse_test > epsilon && iter < max_iter,
        mse_before = mse_test;
        mse_test = 0;
        [_, mse_test] = backprop_batch_step(X_te, y_te, W, alpha);
        mse_test_vec = [mse_test_vec; mse_test];
        [W, mse_train] = backprop_batch_step(X_tr, y_tr, W, alpha);
        mse_train_vec = [mse_train_vec; mse_train];
        
        if mse_test > mse_min,
            counter = counter + 1;
        else,
            mse_min = mse_test;
            counter = 0;
        end
        
        iter = iter + 1;
        
        if show_steps,
            iter
            [mse_train, mse_test]
        end
    end
    
    iter_min = iter - counter
    mse_min
    
end

# hyperparameters
epsilon = 0.1;
alpha = 0.006;
gamma = 50;
max_iter = 20000;

% backpropagation for number of hidden neurons as defined above in ANN architecture
[W_out, mse_train_vec, mse_test_vec] = batch_backpropagation(X_test_bias, X_train_bias, y_test, y_train, W_bias, alpha, epsilon, gamma, 1, max_iter);

% learning curve
plot(mse_train_vec)
hold on
plot(mse_test_vec, 'r')
title('Learning curve')
legend('train set', 'test set')

% feedfoward ann evaluation
output = sum(X_train_bias' * W_out', 2);

% comparison of output and labels
[output, y_train, (y_train - output)]

i = 1;
W_out = [];

% comparison of number of neurons in hidden layer
for O = [1, 2, 3, 5, 10, 30],
    
    neurons_hidden_layer = O
    
    W = rand([O N]).*0.01;
    W_to_store = zeros(size(W));
    W_bias = [ones(size(W,1),1), W];
    
    [W_to_store, mse_train_vec, mse_test_vec] = batch_backpropagation(X_test_bias, X_train_bias, y_test, y_train, W_bias, alpha, epsilon, gamma, 0, max_iter);
    W_out = [W_out; W_to_store];
    
    % learning curve
    figure(i)
    plot(mse_train_vec)
    hold on
    plot(mse_test_vec, 'r')
    title(['ANN with ' num2str(O) ' neurons in hidden layer'])
    legend('train set', 'test set')
    
    i = i+1;
    
end
