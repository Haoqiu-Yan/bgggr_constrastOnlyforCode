function [ytest_fit, train_rmse, train_mse, test_rmse, test_mse, time]=bagging(filedata)
%% ��������
data = load(filedata);  
trainingData=data.train;
testData=data.test;
xtrain=trainingData(:,1:75);
ytrain=trainingData(:,76);
xtest=testData(:,1:75);
ytest=testData(:,76);
%% ģ��ѵ��
tic

desired_i=[];
mse_max=1;
desired_j=[];
result_perfp=[];
A=[];
spread_result=[];
for i = 1:1:4
    perfp=[];
    disp(['��ǰҶ����ĿΪ',num2str(i)])
    for j=10:5:300
        disp(['��ǰѧϰ����ĿΪ',num2str(j)])
        [trainedModel, validationRMSE] = trainRegressionModel_10fold_75(trainingData,i,j);
        kfoldrmse=validationRMSE;
        kfoldmse=kfoldrmse^2;
        disp(['��ǰģ�͵�rmseΪ',num2str(kfoldrmse)]);
        disp(['��ǰģ�͵�mseΪ',num2str(kfoldmse)]);
        
        ytest_fit = trainedModel.predictFcn(xtest);
        ytrain_fit = trainedModel.predictFcn(xtrain);
        train_error=ytrain-ytrain_fit;
        train_mse = mse(train_error);
        train_rmse=sqrt(train_mse);
        test_error=ytest-ytest_fit;
        test_mse = mse(test_error);
        test_rmse=sqrt(test_mse);
        A=[i,j,train_mse,train_rmse,test_mse,test_rmse,kfoldmse,kfoldrmse];
        spread_result=[A;spread_result;];
        
        if kfoldmse<mse_max
            mse_max=kfoldmse;
            desired_i=i;
            desired_j=j;
            tenfoldmodel75=trainedModel;
        end
    end
end
  
disp(['���Ҷ����ĿΪ',num2str(desired_i)])
disp(['���ѧϰ����ĿΪ',num2str(desired_j)])
%% ����Ԥ��
ytest_fit = tenfoldmodel75.predictFcn(xtest);
ytrain_fit = tenfoldmodel75.predictFcn(xtrain);

train_error=ytrain-ytrain_fit;
train_mse = mse(train_error);
train_rmse=sqrt(train_mse);
test_error=ytest-ytest_fit;
test_mse = mse(test_error);
test_rmse=sqrt(test_mse);

disp(['[bagging]ensembleԤ��train��rmse���Ϊ',num2str(train_rmse)]);
disp(['[bagging]ensembleԤ��train��mse���Ϊ',num2str(train_mse)]);
disp(['[bagging]ensembleԤ��test��rmse���Ϊ',num2str(test_rmse)]);
disp(['[bagging]ensembleԤ��test��mse���Ϊ',num2str(test_mse)]);

time=toc;
%% ��Ԥ��ֵд��excel��
% xlswrite('o1_bagging75_10fold_ytest_fit.csv',ytest_fit,'ytest fit');
% b={'train rmse',train_rmse;'train mse',train_mse;
    % 'test_rmse',test_rmse;'test_mse',test_mse;
    % 'time',time};
% xlswrite('o1_bagging75_10fold_ytest_fit.csv', b, 'model mes');
%% ��¼����ʱ��
disp(['bagging������ʱ��Ϊ',time]);
save([filedata, '_bagging_process.mat'])




