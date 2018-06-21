function y = neuron(w, x)
    u = w'*x;
    y = tanh(u);
end
