function [ytest_fit, train_rmse, train_mse, test_rmse, test_mse, train_pll, test_pll, time]=ardgpr(filedata)
%% 加载数据
data = load(filedata);  
trainingData=data.train;
testData=data.test;
xtrain=trainingData(:,1:75);
ytrain=trainingData(:,76);
xtest=testData(:,1:75);
ytest=testData(:,76);
%% 模型训练
tic

mean(xtrain), std(xtrain), mean(ytrain), std(ytrain)
offset = mean(ytrain);
disp('  y = y - offset;        % centre targets around 0')
ytrain = ytrain - offset;

covfunc = {'covSum', {'covSEard','covNoise'}};


logtheta0 = [0;0;0;0;0;  0;0;0;0;0;0;0;0;0;0; 0;0;0;0;0;0;0;0;0;0; 0;0;0;0;0;0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;0;0;0;0;0;0;0;0;0; 0; 0; 0; 0;0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;      0;0;0;0;0;  0; log(0.9)];

[logtheta, fvals, iter] = minimize(logtheta0, 'gpr', -100, covfunc, xtrain, ytrain);

disp(' ')
disp('We now plot the negative marginal likelihood as a function of the')
disp('number of line-searches of the optimization routine.')
% disp(' ');
% disp('Press any key to make the plot.')
% pause;

% plot(fvals)
% hold on
% plot(fvals,'bo')
% ylabel('negative marginal likelihood')
% xlabel('number of line-searches')
% hold off

disp(' ')
disp('We now study the learned hyperparameters:')
disp(' ')
% disp('Press any key to continue')
% pause;

disp(' ')

A=[];
ell_result=[];
for i=1:75
    fprintf(1, 'ell_ %d \t\t%12.6f\n',i,exp(logtheta(i)));
    A=[i,exp(logtheta(i))];
    ell_result=[A;ell_result;];
end
fprintf(1, 'sigma_f \t%12.6f\n',exp(logtheta(76)));
A=[76,exp(logtheta(76))];
ell_result=[A;ell_result;];
fprintf(1, 'sigma_n \t%12.6f\n',exp(logtheta(77 )));
A=[77,exp(logtheta(77))];
ell_result=[A;ell_result;];

% 训练�?
[ytrain_fit, S2_train] = gpr(logtheta, covfunc, xtrain, ytrain, xtrain);
ytrain_fit = ytrain_fit + offset;
train_error = ytrain-ytrain_fit;
train_mse = mse(train_error);
train_rmse = sqrt(train_mse);
train_pll = -0.5*mean(log(2*pi*S2_train)+train_error.^2./S2_train);
% 测试�?
[ytest_fit, S2_test] = gpr(logtheta, covfunc, xtrain, ytrain, xtest);
ytest_fit = ytest_fit + offset;
test_error = ytest-ytest_fit;
test_mse = mse(test_error);
test_rmse = sqrt(test_mse);
test_pll = -0.5*mean(log(2*pi*S2_test)+test_error.^2./S2_test);

fprintf(1,'The test mse is %10.6f\n', test_mse);
fprintf(1,'The test RMSE is %10.6f\n', test_rmse);
fprintf(1,'and the test mean predictive log likelihood is %6.4f.\n', test_pll);
time = toc;

%% 绘制并保存图�?

%% 记录运行时间
disp(['ardgpr的运行时间为',time]);
save([filedata, '_ardgpr_process.mat'])





















