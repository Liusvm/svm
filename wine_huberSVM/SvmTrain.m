function model = SvmTrain(X, y, lambda, gamma, sigma, kernelFunction, eps, max_passes)


if nargin == 5
    kernelFunction = 'gaussianKernel';
    eps = 0.01;
    max_passes = 1000;
elseif nargin == 6
    eps = 0.01;
    max_passes = 1000;
elseif nargin == 7
    eps = 0.01;
    max_passes = 1000;
end


[sampleNumbers, dim] = size(X);
Y = diag(y);
e = ones(sampleNumbers, 1);

if strcmp(kernelFunction, 'gaussianKernel')
    K = gaussianKernel(X, X', sigma);
    u = ones(sampleNumbers+1, 1)*0.5;
    I = eye(sampleNumbers + 1);
elseif strcmp(kernelFunction, 'lineKernel')
    K = X;
    u = ones(dim+1, 1).*0.5;
    I = eye(dim + 1);
end


P = [K, e];

pass = 1;
success = true;
while pass <= max_passes
    u = [u (I/lambda + P'*P)^-1 *P'* Y * (e - max(e - Y*P*u(:, end) - e*gamma, zeros(sampleNumbers, 1)) + max(-e + Y*P*u(:, end) - e*gamma, zeros(sampleNumbers, 1)))];
    pass = pass + 1;
    
    if norm(u(:, end) - u(:, end-1)) < eps
        break;
    end
    
    if pass == max_passes
        success = false;
        fprintf('Function Error: The function is non convergence');
    end
    
end



if success
    model.X = X;
    model.y = y;
    model.kernelFunction = kernelFunction;
    if strcmp(kernelFunction, 'gaussianKernel')
        model.w = u(1:sampleNumbers, end);
        model.b =  u(sampleNumbers+1, end);
        model.sigma = sigma;
    elseif strcmp(kernelFunction, 'lineKernel')
        model.w = u(1:end-1, end);
        model.b =  u(end, end);
    end
    model.u = u;
end

