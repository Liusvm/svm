clc;
close all;
clear all;

[class,Alcohol, Malic, Ash,  Alc,Mag,Tot,Fla,Nonf,Pro,Color,Hue,dilut,Proline]= textread('wine.data', ...
    '%f%f%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',',');


data = [Hue,Alcohol, Malic, Ash, Alc,Mag,Tot,Fla,Nonf,Pro,Color,dilut,Proline,class];

alldata = sortrows(data);
% # of rows
N1 = size(alldata, 1);

% # of columns
N2 = size(alldata,2);




% scaled input variables to zero mean 1 variance
for i = 1:N2-1
    tmp1 = min(alldata(:,i));
    tmp2 = max(alldata(:,i));
    alldata(:,i) = (alldata(:,i)-tmp1)/(tmp2-tmp1);
end




train = [];
test = [];

for i = 1:3
    testindex = [];
    trainindex = [];
    % compute the index
    for j = 1:N1
        if mod(j,5)==mod(i,5)
            testindex = [testindex j];
        else
            trainindex = [trainindex j];
        end
    end
    
    % train data for this partition
    train = alldata(trainindex,:);
    
    % test data for this partition
    test = alldata(testindex,:);

    testsample{1}.data = [];
    testlabel{1}.data= [];
    testsample{2}.data = [];
    testlabel{2}.data= [];
    testsample{3}.data = [];
    testlabel{3}.data = [];
 
    trainsample{1}.data = [];
    trainlabel{1}.data= [];
    trainsample{2}.data = [];
    trainlabel{2}.data = [];
    trainsample{3}.data = [];
    trainlabel{3}.data = [];
 
    
 
  
     for j = 1:size(test,1)
        if test(j,1)<0.31
            testsample{2}.data = [testsample{2}.data; test(j,1:N2-1)];
            if test(j,N2)==2
                testlabel{2}.data = [testlabel{2}.data; 1];
            else
                testlabel{2}.data = [testlabel{2}.data; -1];
            end
        elseif test(j,1)<0.48
           testsample{1}.data = [testsample{1}.data; test(j,1:N2-1)];
            if test(j,N2)==2
                testlabel{1}.data = [testlabel{1}.data; 1];
            else
                testlabel{1}.data = [testlabel{1}.data; -1];
            end
        else
             testsample{3}.data = [ testsample{3}.data; test(j,1:N2-1)];
            if test(j,N2)==2
                testlabel{3}.data = [testlabel{3}.data; 1];
            else
                testlabel{3}.data = [testlabel{3}.data; -1];
            end
        end
        
    end
    
 
    for j = 1:size(train,1)
        if train(j,1)<0.31
            trainsample{2}.data = [trainsample{2}.data; train(j,1:N2-1)];
            if train(j,N2)==2
                trainlabel{2}.data = [trainlabel{2}.data; 1];
            else
                trainlabel{2}.data = [trainlabel{2}.data; -1];
            end          
        elseif train(j,1)<0.48
            trainsample{1}.data = [trainsample{1}.data; train(j,1:N2-1)];
            if train(j,N2)==2
                trainlabel{1}.data = [trainlabel{1}.data; 1];
            else
               trainlabel{1}.data = [trainlabel{1}.data; -1];
            end
        else
            trainsample{3}.data = [trainsample{3}.data; train(j,1:N2-1)];
            if train(j,N2)==2
                trainlabel{3}.data = [trainlabel{3}.data; 1];
            else
                trainlabel{3}.data = [trainlabel{3}.data; -1];
            end
        end
        
    end
    
    train_s = [];
    train_l = [];
    test_s = [];
    test_l = [];
    
     
       train_sample=[trainsample{1}.data;trainsample{2}.data;trainsample{3}.data];
       train_label=[ trainlabel{1}.data; trainlabel{2}.data; trainlabel{3}.data];
       test_sample=[testsample{1}.data;testsample{2}.data;testsample{3}.data];
        test_label=[testlabel{1}.data;testlabel{2}.data;testlabel{3}.data];
    
    for k =1:size(trainsample, 2)
        
        t1 = cputime;
        svm_classifier = SvmTrain(trainsample{k}.data, trainlabel{k}.data, 10, 0.2, 1.0);
        t2 = cputime;
        accuracyGM(k)  = length(find(SvmPredict(svm_classifier, testsample{k}.data) == testlabel{k}.data))/length(testlabel{k}.data);
        GMTime(k) = t2-t1;
        
        t1 = cputime;
        lsvm_classifier = SvmTrain(trainsample{k}.data, trainlabel{k}.data, 10, 1.0, 0.0, 'lineKernel');
        t2= cputime;
        accuracyLM(k)  = length(find(SvmPredict(lsvm_classifier, testsample{k}.data) == testlabel{k}.data))/length(testlabel{k}.data);
        LMTime(k) = t2-t1;
        
        
        train_s = [train_s; trainsample{k}.data];
        train_l = [train_l; trainlabel{k}.data];
        test_s = [test_s; testsample{k}.data];
        test_l = [test_l; testlabel{k}.data];
    end
    
    t1 = cputime;
    svm_classifier = SvmTrain(train_sample, train_label, 10, 0.2, 0.8);
    t2 = cputime;
    accuracySG(i)  = length(find(SvmPredict(svm_classifier,  test_sample) == test_label))/length(test_label);
    SGTime(i) = t2-t1;
    
    t1 = cputime;
    lsvm_classifier = SvmTrain(train_sample, train_label, 10, 0.8, 0.0, 'lineKernel');
    t2 = cputime;
    accuracySL(i) = length(find(SvmPredict(lsvm_classifier, test_sample) == test_label))/length(test_label);
    SLTime(i) = t2-t1;
    
    accuracy_mGSVM(i) = mean(accuracyGM);
    mGSVMTime(i) = sum(GMTime);
    accuracy_mLSVM(i) = mean(accuracyLM);
    mLSVMTime(i) = sum(LMTime);
    
    
    t1 = cputime;
    svm_classifierGMTL = MultiTaskGSVMTrain(trainsample, trainlabel, 0.07, 10, 0.2, 1.0);
    t2 = cputime;
    accuracy_mlt_gsvm(i) =  MultiTaskGSVMPredict(svm_classifierGMTL, testsample, testlabel);
    mlt_gsvm_time(i) = t2-t1;
    
    t1 = cputime;
    svm_classifierMTL = MultiTaskSVMTrain(trainsample, trainlabel, 0.07, 10, 1.0);
    t2 = cputime;
    accuracy_mlt_lsvm(i) =  MultiTaskSVMPredict(svm_classifierMTL, testsample, testlabel);
    mlt_lsvm_time(i) = t2-t1;
    
end

fprintf('The prediction accuracy results: \n');
 fprintf('sLSVM: %f ', mean(accuracySL));
fprintf([setstr(177), ' %f\n'], std(accuracySL));
 fprintf('time %f\n', mean(SLTime));
fprintf('MLineSVM: %f ', mean(accuracy_mLSVM));
fprintf([setstr(177), ' %f\n'], std(accuracy_mLSVM));
fprintf('time %f\n', mean(mLSVMTime));
fprintf('MLTLineSVM: %f ', mean(accuracy_mlt_lsvm));
fprintf([setstr(177), ' %f\n'], std(accuracy_mlt_lsvm));
fprintf('time %f\n', mean(mlt_lsvm_time));
fprintf('------------------------------------------------\n');
 fprintf('sGSVM: %f ', mean(accuracySG));
 fprintf([setstr(177), ' %f\n'], std(accuracySG));
 fprintf('time %f\n', mean(SGTime));
fprintf('mGSVM:    %f ', mean(accuracy_mGSVM));
fprintf([setstr(177), ' %f\n'], std(accuracy_mGSVM));
fprintf('time %f\n', mean(mGSVMTime));
fprintf('MLTGSVM: %f ', mean(accuracy_mlt_gsvm));
fprintf([setstr(177), ' %f\n'], std(accuracy_mlt_gsvm));
fprintf('time %f\n', mean(mlt_gsvm_time));