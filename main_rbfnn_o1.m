%运行的主函数：对每个数据集调用五个算法
clear;
close all;
%Start Parallel Compute
core_number=35;            %想要调用的处理器个数
parpool('local',core_number);
ProjectDir = pwd;
% ProjectDir='/home/lab421/MATLAB/projects/YanHaoqiuBgggr_constrastOnlyforCode';
% SysPathSeperator='\';
SysPathSeperator='/';
AutomsDir='atoms';
AutomsPath=[ProjectDir, SysPathSeperator, AutomsDir];
algorithm='rbfnn';
CsvoutputDir=[algorithm, '_output'];
CsvoutputPath=[ProjectDir, SysPathSeperator, CsvoutputDir];
diary([ProjectDir, SysPathSeperator, 'o1_rbfnn_log4.txt'])
%cd [ProjectDir, SysPathSeperator, RootDir]
AutomList=dir(AutomsPath); %获得o1,c2,...
for j=1:length(AutomList)
    %跳过fileList结构体中前两行，并筛掉不是文件夹的
    if strcmp(AutomList(j).name,'.')==1 || strcmp(AutomList(j).name,'..')==1 || ~AutomList(j).isdir
        continue;
    end
    autom=AutomList(j).name;
	if strcmp(autom,'o1') == 0
        disp(autom);
        continue 
	end
    DataSetsList=dir([AutomsPath, SysPathSeperator, autom, SysPathSeperator, '*.mat']);
    FitsTable=table;
    StatisticsTable=table;
    parfor i=1:length(DataSetsList)
        %跳过fileList结构体中前两行
        if strcmp(DataSetsList(i).name,'.')==1 || strcmp(DataSetsList(i).name,'..')==1
            continue;
        end
        DataSet=DataSetsList(i).name;
        DataSetNoExtn=strsplit(DataSet,'.');
        DataSetNoExtn=DataSetNoExtn{1};
        DataSetPath=[AutomsPath, SysPathSeperator, autom, SysPathSeperator, DataSetsList(i).name];
        disp(['DataSet is: ', DataSetPath])
        [ytest_fit, train_rmse, train_mse, test_rmse, test_mse, time]=rbfnn(DataSetPath);

        %将预测值存入table
        %eval(['FitsTable.', DataSetNoExtn, '=ytest_fit;']);
        ytest_fit_cell = num2cell(ytest_fit');
        FitsTable(i,:)=ytest_fit_cell;
           
        %将mse, time存入table
        StatisticsTable(i, :)={train_rmse, train_mse, test_rmse,...
            test_mse, time, DataSetNoExtn};
    end
    % delete(gcp('nocreate')) % 关闭并行池
%     %判断有没有该原子的文件夹
    if ~isfolder([CsvoutputPath, SysPathSeperator, autom])
        mkdir([CsvoutputPath, SysPathSeperator, autom])
    end
    FitsCsvName=['fit_', autom, '_', algorithm, '.csv'];
    FitsCsvPath=[CsvoutputPath, SysPathSeperator, autom, SysPathSeperator, FitsCsvName];

    StatisticsTable.Properties.VariableNames={'train rmse', 'train mse',...
        'test_rmse', 'test_mse', 'time', 'data_set_name'};
    StatisticsCsvName=['statistics_', autom, '_', algorithm, '.csv'];
    StatisticsCsvPath=[CsvoutputPath, SysPathSeperator, autom, SysPathSeperator, StatisticsCsvName];

    writetable(FitsTable, FitsCsvPath, 'WriteVariableNames', true)
    writetable(StatisticsTable, StatisticsCsvPath, 'WriteRowNames', true, 'WriteVariableNames', true)
end
diary off