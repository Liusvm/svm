clc;
close all;
clear all;

[class,Alcohol, Malic, Ash,  Alc,Mag,Tot,Fla,Nonf,Pro,Color,Hue,dilut,Proline]= textread('wine.data', ...
    '%f%f%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',',');


data = [Hue,Alcohol, Malic, Ash, Alc,Mag,Tot,Fla,Nonf,Pro,Color,dilut,Proline,class];

% order all data by increasing age
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

finalaccuracy_mg = [];
finalaccuracy_s = [];
finalaccuracy_sl = [];
finalaccuracy_ml = [];
finalaccuracy_mlt = [];
finalaccuracy_mltg = [];


Lambda_mSVM_1 = zeros(3,1);
Gamma_mSVM_1 = zeros(3,1);
Sigma_mSVM_1 = zeros(3,1);

Lambda_mSVM_2 = zeros(3,1);
Gamma_mSVM_2 = zeros(3,1);
Sigma_mSVM_2 = zeros(3,1);

Lambda_mSVM_3 = zeros(3,1);
Gamma_mSVM_3 = zeros(3,1);
Sigma_mSVM_3 = zeros(3,1);

Lambda_mlSVM_1 = zeros(3,1);
Gamma_mlSVM_1 = zeros(3,1);
Sigma_mlSVM_1 = zeros(3,1);

Lambda_mlSVM_2 = zeros(3,1);
Gamma_mlSVM_2 = zeros(3,1);
Sigma_mlSVM_2 = zeros(3,1);

Lambda_mlSVM_3 = zeros(3,1);
Gamma_mlSVM_3 = zeros(3,1);
Sigma_mlSVM_3 = zeros(3,1);

Lambda_sSVM = zeros(3,1);
Gamma_sSVM = zeros(3,1);
Sigma_sSVM = zeros(3,1);

Lambda_slSVM = zeros(3,1);
Gamma_slSVM = zeros(3,1);

Lambda_mltSVM = zeros(3,1);
Gamma_mltSVM = zeros(3,1);
C1_mltSVM = zeros(3,1);

Lambda_mltGSVM = zeros(3,1);
Gamma_mltGSVM = zeros(3,1);
Sigma_mltGSVM = zeros(3,1);
C2_mltGSVM = zeros(3,1);

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
    
    train = alldata(trainindex,:);
    

    test = alldata(testindex,:);
    
    % construct the test data
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
    

    Lambda = [2^-5  2^-3  2^-1  2^1  2^3  2^5 ];
         Sigma= [2^-5  2^-3  2^-1  2^1  2^3  2^5];
         Gamma= [0.1 0.3 0.5 0.7 0.9];
         C1=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
         C2=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
     
       train_sample=[trainsample{1}.data;trainsample{2}.data;trainsample{3}.data];
       train_label=[ trainlabel{1}.data; trainlabel{2}.data; trainlabel{3}.data];
       test_sample=[testsample{1}.data;testsample{2}.data;testsample{3}.data];
        test_label=[testlabel{1}.data;testlabel{2}.data;testlabel{3}.data];
       
      
         for p1  = 1:size(Lambda, 2)
            lambda = Lambda(p1);
            for p2 = 1:size(Gamma,2)
                gamma = Gamma(p2);
                
                svm_classifier = SvmTrain( train_sample ,train_label, lambda, gamma, 0.1, 'lineKernel');
                accuracy_sl(i, j, p1, p2) = length(find(SvmPredict(svm_classifier, test_sample) == test_label))/length(test_label);
               
                
                lsvm_classifier_1 = SvmTrain(trainsample{1}.data,trainlabel{1}.data, lambda, gamma, 0.1, 'lineKernel');
                accuracyL_1(i, j, p1, p2)  = length(find(SvmPredict(lsvm_classifier_1, testsample{1}.data) == testlabel{1}.data))/length(testlabel{1}.data);
                
                lsvm_classifier_2 = SvmTrain(trainsample{2}.data, trainlabel{2}.data, lambda, gamma, 0.1, 'lineKernel');
                accuracyL_2(i, j, p1, p2)  = length(find(SvmPredict(lsvm_classifier_2, testsample{2}.data) == testlabel{2}.data))/length(testlabel{2}.data);
                
                lsvm_classifier_3 = SvmTrain(trainsample{3}.data, trainlabel{3}.data, lambda, gamma, 0.1, 'lineKernel');
                accuracyL_3(i, j, p1, p2)  = length(find(SvmPredict(lsvm_classifier_3, testsample{3}.data) == testlabel{3}.data))/length(testlabel{3}.data);
                
              
                
                for p3 = 1:size(Sigma,2)
                    sigma = Sigma(p3);
                    svm_classifier_1 = SvmTrain(trainsample{1}.data, trainlabel{1}.data, lambda, gamma, sigma);
                    accuracy_1(i, j, p1, p2, p3)  = length(find(SvmPredict(svm_classifier_1, testsample{1}.data) == testlabel{1}.data))/length(testlabel{1}.data);
                    
                    svm_classifier_2 = SvmTrain(trainsample{2}.data, trainlabel{2}.data, lambda, gamma, sigma);
                    accuracy_2(i, j, p1, p2, p3)  = length(find(SvmPredict(svm_classifier_2, testsample{2}.data) == testlabel{2}.data))/length(testlabel{2}.data);
                    
                    svm_classifier_3 = SvmTrain(trainsample{3}.data, trainlabel{3}.data, lambda, gamma, sigma);
                    accuracy_3(i, j, p1, p2, p3)  = length(find(SvmPredict(svm_classifier_3, testsample{3}.data) == testlabel{3}.data))/length(testlabel{3}.data);
                    
                    
                   svm_classifier = SvmTrain(train_sample ,train_label, lambda, gamma, sigma);
                   accuracy_s(i, j, p1, p2, p3) = length(find(SvmPredict(svm_classifier, test_sample) == test_label))/length(test_label);
                   
                   for p5 = 1:size(C2,2)
                        c2 = C2(p5); 
                    svm_classifierGMTL = MultiTaskGSVMTrain(trainsample, trainlabel, c2, lambda, gamma, sigma);
                    accuracy_mlt_gsvm(i, j, p1, p2, p3,p5) = MultiTaskGSVMPredict(svm_classifierGMTL, testsample, testlabel);
                end
            end
                
              for p4 = 1:size(C1,2)
               c1 = C1(p4);
                svm_classifierMTL = MultiTaskSVMTrain(trainsample,trainlabel,c1,lambda, gamma);
                accuracy_mlt(i, j, p1, p2,p4) =  MultiTaskSVMPredict(svm_classifierMTL, testsample, testlabel);
        end
         end
    end
    
    % select optimal parameters
    innersvmm1_maxaccuracy(i) = 0;
    opt_Lambda1(i) = 0.001;
    opt_Gamma1(i) = 10;
    opt_Sigma1(i) = 0.5;
    
    innersvmm2_maxaccuracy(i) = 0;
    opt_Lambda2(i) = 0.001;
    opt_Gamma2(i) = 10;
    opt_Sigma2(i) = 0.5;
    
    innersvmm3_maxaccuracy(i) = 0;
    opt_Lambda3(i) = 0.001;
    opt_Gamma3(i) = 10;
    opt_Sigma3(i) = 0.5;
    
   
    
    innersvmmline1_maxaccuracy(i) = 0;
    opt_LambdaL1(i) = 0.001;
    opt_GammaL1(i) = 10;
    
    innersvmmline2_maxaccuracy(i) = 0;
    opt_LambdaL2(i) = 0.001;
    opt_GammaL2(i) = 10;
    
    innersvmmline3_maxaccuracy(i) = 0;
    opt_LambdaL3(i) = 0.001;
    opt_GammaL3(i) = 10;
    
    
    
    
    innersvms_maxaccuracy(i) = 0;
    opt_sLambda(i) = 0.1;
    opt_sGamma(i) = 10;
    opt_sSigma(i) = 0.5;
    
    innersvmsl_maxaccuracy(i) = 0;
    opt_slLambda(i) = 0.1;
    opt_slGamma(i) = 10;
    
    innersvmmlt_maxaccuracy(i) = 0;
    opt_mltLambda(i) = 0.1;
    opt_mltGamma(i) = 10;
     opt_mltC1(i) = 0.1;
     
    innersvmmltg_maxaccuracy(i) = 0;
    opt_mltgLambda(i) = 0.1;
    opt_mltgGamma(i) = 10;
    opt_mltgSigma(i) = 0.5;
     opt_mltC2(i) = 0.1;
     
    for p1 = 1:size(Lambda, 2)
        for p2 = 1:size(Gamma,2)
            
            
            acc_slsvml1 = mean(accuracyL_1(i, :, p1, p2));
            if innersvmmline1_maxaccuracy(i) < acc_slsvml1
                innersvmmline1_maxaccuracy(i) = acc_slsvml1;
                opt_LambdaL1(i) = Lambda(p1);
                opt_GammaL1(i) = Gamma(p2);
            end
            
            acc_slsvml2 = mean(accuracyL_2(i, :, p1, p2));
            if innersvmmline2_maxaccuracy(i) < acc_slsvml2
                innersvmmline2_maxaccuracy(i) = acc_slsvml2;
                opt_LambdaL2(i) = Lambda(p1);
                opt_GammaL2(i) = Gamma(p2);
            end
            
            acc_slsvml3 = mean(accuracyL_3(i, :, p1, p2));
            if innersvmmline3_maxaccuracy(i) < acc_slsvml3
                innersvmmline3_maxaccuracy(i) = acc_slsvml3;
                opt_LambdaL3(i) = Lambda(p1);
                opt_GammaL3(i) = Gamma(p2);
            end
             
            
            
            acc_slsvm = mean(accuracy_sl(i, :, p1, p2));
            if innersvmsl_maxaccuracy(i) < acc_slsvm
                innersvmsl_maxaccuracy(i) = acc_slsvm;
                opt_slLambda(i) = Lambda(p1);
                opt_slGamma(i) = Gamma(p2);
            end
            
          
            
            for p3 = 1:size(Sigma,2)
                
                acc_svm1 = mean(accuracy_1(i, :, p1, p2, p3));
                if innersvmm1_maxaccuracy(i) < acc_svm1
                    innersvmm1_maxaccuracy(i) = acc_svm1;
                    opt_Lambda1(i) = Lambda(p1);
                    opt_Gamma1(i) = Gamma(p2);
                    opt_Sigma1(i) = Sigma(p3);
                end
                
                acc_svm2 = mean(accuracy_2(i, :, p1, p2, p3));
                if innersvmm2_maxaccuracy(i) < acc_svm2
                    innersvmm2_maxaccuracy(i) = acc_svm2;
                    opt_Lambda2(i) = Lambda(p1);
                    opt_Gamma2(i) = Gamma(p2);
                    opt_Sigma2(i) = Sigma(p3);
                end
                
                acc_svm3 = mean(accuracy_3(i, :, p1, p2, p3));
                if innersvmm3_maxaccuracy(i) < acc_svm3
                    innersvmm3_maxaccuracy(i) = acc_svm3;
                    opt_Lambda3(i) = Lambda(p1);
                    opt_Gamma3(i) = Gamma(p2);
                    opt_Sigma3(i) = Sigma(p3);
                end
                
                
                
                
                
                acc_ssvm = mean(accuracy_s(i, :, p1, p2, p3));
                if innersvms_maxaccuracy(i) < acc_ssvm
                    innersvms_maxaccuracy(i) = acc_ssvm;
                    opt_sLambda(i) = Lambda(p1);
                    opt_sGamma(i) = Gamma(p2);
                    opt_sSigma(i) = Sigma(p3);
                end
                
                 for p5 = 1:size(C2,2)
                
                acc_gsvmmlt = mean(accuracy_mlt_gsvm(i, :, p1, p2, p3,p5));
                if innersvmmltg_maxaccuracy(i) < acc_gsvmmlt
                    innersvmmltg_maxaccuracy(i) = acc_gsvmmlt;
                    opt_mltgLambda(i) = Lambda(p1);
                    opt_mltgGamma(i) = Gamma(p2);
                    opt_mltgSigma(i) = Sigma(p3);
                    opt_mltgC2(i) =C2(p5);
                end
                 end
            end
            
             for p4 = 1:size(C1,2)
            
              acc_ssvmlt = mean(accuracy_mlt(i, :, p1, p2,p4));
            if innersvmmlt_maxaccuracy(i) < acc_ssvmlt
                innersvmmlt_maxaccuracy(i) = acc_ssvmlt;
                opt_mltLambda(i) = Lambda(p1);
                opt_mltGamma(i) = Gamma(p2);
                opt_mltgC1(i) =C1(p4);
            end
             end
            
        end
    end
     lambda = opt_Lambda1(i);
    gamma = opt_Gamma1(i);
    sigma = opt_Sigma1(i);
    Lambda_mSVM_1(i, 1) = opt_Lambda1(i);
    Gamma_mSVM_1(i, 1) = opt_Gamma1(i);
    Sigma_mSVM_1(i, 1) = opt_Sigma1(i);
    svm_classifier1 = SvmTrain(trainsample{1}.data, trainlabel{1}.data, lambda, gamma, 0.1, 'lineKernel');
    svm_pred1 = length(find(SvmPredict(svm_classifier1, testsample{1}.data) == testlabel{1}.data))/length(testlabel{1}.data);
    
    lambda = opt_Lambda2(i);
    gamma = opt_Gamma2(i);
    sigma = opt_Sigma2(i);
    Lambda_mSVM_2(i, 1) = opt_Lambda2(i);
    Gamma_mSVM_2(i, 1) = opt_Gamma2(i);
    Sigma_mSVM_2(i, 1) = opt_Sigma2(i);
    svm_classifier2 = SvmTrain(trainsample{2}.data, trainlabel{2}.data, lambda, gamma, 0.1, 'lineKernel');
    svm_pred2 = length(find(SvmPredict(svm_classifier2, testsample{2}.data) == testlabel{2}.data))/length(testlabel{2}.data);
    
    lambda = opt_Lambda3(i);
    gamma = opt_Gamma3(i);
    sigma = opt_Sigma3(i);
    Lambda_mSVM_3(i, 1) = opt_Lambda3(i);
    Gamma_mSVM_3(i, 1) = opt_Gamma3(i);
    Sigma_mSVM_3(i, 1) = opt_Sigma3(i);
    svm_classifier3 = SvmTrain(trainsample{3}.data, trainlabel{3}.data, lambda, gamma, 0.1, 'lineKernel');
    svm_pred3 = length(find(SvmPredict(svm_classifier3, testsample{3}.data) == testlabel{3}.data))/length(testlabel{3}.data);
    
    
    finalaccuracy_ml = [finalaccuracy_ml; (svm_pred1*size(testlabel{1}.data, 1)...
        +svm_pred2*size(testlabel{2}.data, 1)+svm_pred3*size(testlabel{3}.data, 1) )/(size(testlabel{1}.data, 1)+...
        size(testlabel{2}.data, 1)+size(testlabel{3}.data, 1))];
    
    
    
    lambda = opt_Lambda1(i);
    gamma = opt_Gamma1(i);
    sigma = opt_Sigma1(i);
    Lambda_mSVM_1(i, 1) = opt_Lambda1(i);
    Gamma_mSVM_1(i, 1) = opt_Gamma1(i);
    Sigma_mSVM_1(i, 1) = opt_Sigma1(i);
    svm_classifier1 = SvmTrain(trainsample{1}.data, trainlabel{1}.data, lambda, gamma, sigma);
    svm_pred1 = length(find(SvmPredict(svm_classifier1, testsample{1}.data) == testlabel{1}.data))/length(testlabel{1}.data);
    
    lambda = opt_Lambda2(i);
    gamma = opt_Gamma2(i);
    sigma = opt_Sigma2(i);
    Lambda_mSVM_2(i, 1) = opt_Lambda2(i);
    Gamma_mSVM_2(i, 1) = opt_Gamma2(i);
    Sigma_mSVM_2(i, 1) = opt_Sigma2(i);
    svm_classifier2 = SvmTrain(trainsample{2}.data, trainlabel{2}.data, lambda, gamma, sigma);
    svm_pred2 = length(find(SvmPredict(svm_classifier2, testsample{2}.data) == testlabel{2}.data))/length(testlabel{2}.data);
    
    lambda = opt_Lambda3(i);
    gamma = opt_Gamma3(i);
    sigma = opt_Sigma3(i);
    Lambda_mSVM_3(i, 1) = opt_Lambda3(i);
    Gamma_mSVM_3(i, 1) = opt_Gamma3(i);
    Sigma_mSVM_3(i, 1) = opt_Sigma3(i);
    svm_classifier3 = SvmTrain(trainsample{3}.data, trainlabel{3}.data, lambda, gamma, sigma);
    svm_pred3 = length(find(SvmPredict(svm_classifier3, testsample{3}.data) == testlabel{3}.data))/length(testlabel{3}.data);
    
    
    
    finalaccuracy_mg = [finalaccuracy_mg; (svm_pred1*size(testlabel{1}.data, 1)...
        +svm_pred2*size(testlabel{2}.data, 1)+svm_pred3*size(testlabel{3}.data, 1) )/(size(testlabel{1}.data, 1)+...
        size(testlabel{2}.data, 1)+size(testlabel{3}.data, 1))];
    
    
    lambda = opt_sLambda(i);
    Lambda_sSVM(i,1) = opt_sLambda(i);
    gamma = opt_sGamma(i);
    Gamma_sSVM(i,1) = opt_sGamma(i);
    sigma = opt_sSigma(i);
    Sigma_sSVM(i,1) = opt_sSigma(i);
    svm_classifier = SvmTrain(train_sample,train_label, lambda, gamma, sigma);
    finalaccuracy_s = [finalaccuracy_s; length(find(SvmPredict(svm_classifier, test_sample) == test_label))/length(test_label)];
    
    
    lambda = opt_mltgLambda(i);
    Lambda_mltGSVM(i,1) = opt_mltgLambda(i);
    gamma = opt_mltgGamma(i);
    Gamma_mltGSVM(i,1) = opt_mltgGamma(i);
    sigma = opt_mltgSigma(i);
    Sigma_mltGSVM(i,1) = opt_mltgSigma(i);
     c2 = opt_mltgC2(i);
    C2_mltGSVM(i,1) = opt_mltgC2(i);
    svm_classifierGMTL = MultiTaskGSVMTrain(trainsample,trainlabel,c2, lambda, gamma, sigma);
    finalaccuracy_mltg = [finalaccuracy_mltg; MultiTaskGSVMPredict(svm_classifierGMTL, testsample, testlabel)];
    
    lambda = opt_slLambda(i);
    Lambda_slSVM(i,1) = opt_slLambda(i);
    gamma = opt_slGamma(i);
    Gamma_slSVM(i,1) = opt_slGamma(i);
    svm_classifier = SvmTrain(train_sample,train_label, lambda, gamma, 0.1, 'lineKernel');
    finalaccuracy_sl = [finalaccuracy_sl; length(find(SvmPredict(svm_classifier, test_sample) == test_label))/length(test_label)];
    
    
    
    
    lambda = opt_mltLambda(i);
    Lambda_mltSVM(i,1) = opt_mltLambda(i);
    gamma = opt_mltGamma(i);
    Gamma_mltSVM(i,1) = opt_mltGamma(i);
     c1 = opt_mltC1(i);
    C1_mltSVM(i,1) = opt_mltC1(i);
    svm_classifiermlt =  MultiTaskSVMTrain(trainsample,trainlabel, c1,lambda, gamma);
    finalaccuracy_mlt = [finalaccuracy_mlt; MultiTaskSVMPredict(svm_classifiermlt, testsample, testlabel)];
    
end   

fprintf('The prediction accuracy results: \n');
fprintf('sLSVM: %f ', mean(finalaccuracy_sl));
fprintf([setstr(177), ' %f\n'], std(finalaccuracy_sl));
fprintf('MLineSVM: %f ', mean(finalaccuracy_ml));
fprintf([setstr(177), ' %f\n'], std(finalaccuracy_ml));
fprintf('MLTLineSVM: %f ', mean(finalaccuracy_mlt));
fprintf([setstr(177), ' %f\n'], std(finalaccuracy_mlt));
fprintf('sGSVM: %f ', mean(finalaccuracy_s));
fprintf([setstr(177), ' %f\n'], std(finalaccuracy_s));
fprintf('mGSVM:    %f ', mean(finalaccuracy_mg));
fprintf([setstr(177), ' %f\n'], std(finalaccuracy_mg));
fprintf('MLTGSVM: %f ', mean(finalaccuracy_mltg));
fprintf([setstr(177), ' %f\n'], std(finalaccuracy_mltg));

