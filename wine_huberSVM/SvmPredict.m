function pred = SvmPredict(model, x)

if strcmp(model.kernelFunction, 'gaussianKernel')
    pred = sign(gaussianKernel(x, model.X',model.sigma) * model.w + model.b);
elseif strcmp(model.kernelFunction, 'lineKernel')
    pred = sign(x*model.w + model.b);
end

end

