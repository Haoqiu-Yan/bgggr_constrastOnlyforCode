function [ytest_fit, train_rmse, train_mse, test_rmse, test_mse, time]=ardgpr(filedata)
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

desired_i=[];
mse_max=1;
A=[];
spread_result=[];

for i=1:1:100
     % disp(['当前循环为',num2str(i)])
     [trainedModel, validationRMSE] = trainRegressionModel_10fold_75_SE(trainingData);
     cur_rmse=validationRMSE;
     cur_mse=cur_rmse^2;
     disp(['当前模型的rmse为',num2str(cur_rmse)]);
     disp(['当前模型的mse为',num2str(cur_mse)]);
     
     ytrain_fit = trainedModel.predictFcn(xtrain);
     train_error=ytrain-ytrain_fit;
     train_mse = mse(train_error);
     train_rmse=sqrt(train_mse);
     ytest_fit = trainedModel.predictFcn(xtest);
     test_error=ytest-ytest_fit;
     test_mse = mse(test_error);
     test_rmse=sqrt(test_mse);
     A=[i,train_mse,train_rmse,test_mse,test_rmse,cur_mse,cur_rmse];
     spread_result=[A;spread_result];
        
     if test_mse<mse_max
         mse_max=test_mse;
          % desired_i=i;
         tenfoldmodel75=trainedModel;
     end
end
disp(['第',num2str(desired_i),'循环效果最佳'])

% toc
%% 数据预测
ytrain_fit = tenfoldmodel75.predictFcn(xtrain);
train_error=ytrain-ytrain_fit;
train_mse = mse(train_error);
train_rmse=sqrt(train_mse);

ytest_fit = tenfoldmodel75.predictFcn(xtest);
test_error=ytest-ytest_fit;
test_mse = mse(test_error);
test_rmse=sqrt(test_mse);

disp(['GPR预测train的rmse误差为',num2str(train_rmse)]);
disp(['GPR预测train的mse误差为',num2str(train_mse)]);
disp(['GPR预测test的rmse误差为',num2str(test_rmse)]);
disp(['GPR预测test的mse误差为',num2str(test_mse)]);

time=toc;
%% 将预测值写入excel中
%load('reo1_gpr_nofold.mat')
% xlswrite('o1_gpr75_10fold_ytest_fit.xlsx',ytest_fit,'ytest_fit');
% b={'train rmse',train_rmse;'train mse',train_mse;
    % 'test_rmse',test_rmse;'test_mse',test_mse;
    % 'time',time};
% xlswrite('o1_gpr75_10fold_ytest_fit.xlsx', b, 'model mes');
%% 记录运行时间
disp(['gpr的运行时间为',time]);
save([filedata, '_gpr_process.mat'])





