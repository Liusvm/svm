function pred = MultiTaskSVMPredict(model, X, y)

pred = [];
for i =1:model.tasks
    f{i} = sign(X{i}.data * (model.w0 + model.u{i}));
    pred = [ pred length(find(f{i} == y{i}.data))/length(y{i}.data)];
end
pred = mean(pred);

end
