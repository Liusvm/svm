function pred = MultiTaskGSVMPredict(model, X, y, i)


if nargin == 4
    pred = sign(gaussianKernel(X, model.X{i}.data', model.sigma) * (model.w0 + model.u{i}));
else
    pred = []; 
    for i =1:model.tasks
        f{i} = sign(gaussianKernel(X{i}.data, model.X{i}.data', model.sigma) * (model.w0 + model.u{i}));
        pred = [ pred length(find(f{i} == y{i}.data))/length(y{i}.data)];
    end
    pred = mean(pred);
end
end
