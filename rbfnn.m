function [ytest_fit, train_rmse, train_mse, test_rmse, test_mse, time]=rbfnn(filedata)
%% 加载数据
data = load(filedata);  
trainingData=data.train;
testData=data.test;
xtrain=trainingData(:,1:75);
ytrain=trainingData(:,76);
xtest=testData(:,1:75);
ytest=testData(:,76);
errorlimit = 0.0001;

%% 模型训练
tic

desired_spread=[];
mse_max=1;
desired_input=[];
desired_output=[];
result_perfp=[];
A=[];
spread_result=[];
indices=crossvalind('Kfold',length(xtrain),10);
k=1;
for i = 1:10
    perfp=[];
    disp(['以下为第',num2str(i),'次交叉验证结果'])
    test = (indices == i); train = ~test;
    p_cv_train=xtrain(train,:)';
    t_cv_train=ytrain(train,:)';
    p_cv_test=xtrain(test,:)';
    t_cv_test=ytrain(test,:)';
    for spread=20:5:500
        net=newrb(p_cv_train,t_cv_train,errorlimit,spread,500,100);
        disp(['当前spread值为', num2str(spread)]);
        test_Out=sim(net,p_cv_test);
        error=(t_cv_test-test_Out)';
        foldmse=mse(error);
        disp(['当前网络的mse为',num2str(mse(error))]);
        test_Out2=sim(net,xtest');
        error2=(ytest'-test_Out2)';
        testmse=mse(error2);
        train_Out=sim(net,xtrain');
        error1=(ytrain'-train_Out)';
        trainmse=mse(error1);
        A=[i,spread,testmse,trainmse,foldmse];
% foldmse表示当前折使用的测试集的误差
        spread_result=[A;spread_result;];
        perfp=[perfp mse(error)];
        if mse(error)<mse_max
            mse_max=mse(error);
            desired_spread=spread;
            desired_input=p_cv_train;
            desired_output=t_cv_train;
            a=i;
        end
        k=k+1;
    end
    result_perfp(i,:)=perfp;
end
disp(['最佳spread值为',num2str(desired_spread)])
disp(['最佳a值为',num2str(a)])
%% 数据预测
net= newrb(desired_input,desired_output,errorlimit,desired_spread,500,100);
xtrain_t = (xtrain)';
xtest_t = (xtest)';
ytrain_fit = sim(net,xtrain_t)'; 
ytest_fit = sim(net,xtest_t)';

train_error=ytrain-ytrain_fit;
train_mse = mse(train_error);
train_rmse=sqrt(train_mse);

test_error=ytest-ytest_fit;
test_mse = mse(test_error);
test_rmse=sqrt(test_mse);

time = toc;
%% 将预测值写入excel中
% xlswrite('o1_rbf75_10fold_ytest_fit.xlsx',ytest_fit,'ytest_fit');
% b={'train rmse',train_rmse;'train mse',train_mse;
    % 'test_rmse',test_rmse;'test_mse',test_mse;
    % 'time',time};
% xlswrite('o1_rbf75_10fold_ytest_fit.xlsx', b, 'model mes');
%% 记录运行时间
disp(['rbf的运行时间为',time]);
save([filedata, '_rbf_process.mat'])

