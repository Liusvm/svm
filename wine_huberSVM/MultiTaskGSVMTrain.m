function model = MultiTaskGSVMTrain(X, y, C, lambda, gamma, sigma)

tasks = size(X, 2);

sampleNumbers  = 0;
for i = 1:tasks
    sampleNumbers = max(sampleNumbers, size(X{i}.data, 1));
end

for i = 1:tasks
    for j = 1:(sampleNumbers -  size(y{i}.data, 1))
        y{i}.data = [y{i}.data; y{i}.data(end, :)];
        X{i}.data = [X{i}.data; X{i}.data(end, :)];
    end
    e{i} = ones(sampleNumbers, 1);
    Y{i} = diag(y{i}.data);
    I{i} = eye(sampleNumbers);
    K{i} = gaussianKernel(X{i}.data, X{i}.data', sigma);
    U{i} = ones(sampleNumbers, 1)*0.5;
end

pass1 = 1;
max_passes = 2000;
delta = 0.1;

W0 = zeros(sampleNumbers, 1);
while pass1 <= max_passes
    w0 = zeros(sampleNumbers, 1);
    for i = 1:tasks
        pass2 = 1;
        while pass2 <= max_passes
            
            U{i} = [ U{i} ((I{i}*C/tasks + lambda*K{i}'*K{i})^-1) * ...
                (lambda * K{i}'*Y{i}*e{i} - lambda*K{i}'*K{i}*W0(:, end) - ...
                lambda * K{i}'*Y{i} *( ...
                max((e{i} - Y{i}*K{i}*(W0(:, end) + U{i}(:, end))-e{i}*gamma), zeros(sampleNumbers,1)) - ...
                max((-e{i} + Y{i}*K{i}*(W0(:, end) + U{i}(:, end))-e{i}*gamma), zeros(sampleNumbers,1))))];
            
            pass2 = pass2 + 1;
            
            if norm(U{i}(:, end) - U{i}(:, end-1)) < delta
                break;
            end
            
            if pass2 == max_passes
                fprintf(['Function Error: The GSVM function is non convergence {tasks: %i} \n'], i);
            end
        end
        w0 = w0 + U{i}(:, end);
    end
    
    pass1 = pass1 + 1;
    W0 = [W0 (C/tasks)*w0];
    
    if abs(W0(:, end)) - abs(W0(:, end-1)) < delta*5
        break;
    end
    
    if pass1 == max_passes
        fprintf('Function Error: The GSVM w0 is non convergence \n');
    end
    
    
end



model.X = X;
model.y = y;
model.U = U;
model.w0 = W0(:, end);
for i = 1:tasks
    model.u{i} = U{i}(:, end);
end
model.tasks = tasks;
model.sigma = sigma;


end

