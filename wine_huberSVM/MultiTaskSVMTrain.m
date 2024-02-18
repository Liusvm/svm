function model = MultiTaskSVMTrain(X, y, C, lambda, gamma)

tasks = size(X, 2);
dim = size(X{1}.data, 2);
delta = 0.1;

for i = 1:tasks
    e{i} = ones(size(X{i}.data, 1), 1);
    Y{i} = diag(y{i}.data);
    I{i} = eye(dim);
    G{i} = X{i}.data;
    U{i} = ones(dim, 1).*0.5;
end


pass1 = 1;
max_passes = 2000; 
W0 = zeros(dim, 1);
while pass1 < max_passes
    w0 = zeros(dim, 1);
    for i = 1:tasks
        pass2 = 1;
        while pass2 < max_passes
            sampleNumbers = size(X{i}.data, 1);
            U{i} = [U{i} (C*I{i}/tasks + lambda*G{i}'*G{i})^-1 * ...
                (lambda*G{i}'*Y{i}* e{i} - lambda*G{i}'*G{i}*W0(:, end)...
                - lambda*G{i}'*Y{i}*(...
                max((e{i} - Y{i}*G{i}*(W0(:, end) + U{i}(:, end)) - e{i}*gamma), zeros(sampleNumbers, 1))...
                - max((-e{i} + Y{i}*G{i}*(W0(:, end) + U{i}(:, end)) - e{i}*gamma), zeros(sampleNumbers, 1))))];
            
            pass2 = pass2 + 1;
            
            if norm(U{i}(:, end) - U{i}(:, end-1)) < delta
                success = true;
                break;
            end
            if pass2 == max_passes
                success = false;
                fprintf(['Function Error: The function is non convergence tasks: %i \n'], i);
            end
        end
        w0 = w0 + U{i}(:, end);
    end
    pass1 = pass1 + 1;
    W0 = [W0 (C/tasks)*w0];
    
    if abs(W0(:, end)) - abs(W0(:, end-1)) < delta
        break;
    end
    
    if pass1 == max_passes
        success = false;
        fprintf('Function Error: The w0 is non convergence \n');
    end
end




model.X = X;
model.y = y;
model.U = U;
model.w0 = W0(:, end);
model.tasks = tasks;
for i = 1:tasks
    model.u{i} = U{i}(:, end);
end

end

