%���е�����������ÿ�����ݼ���������㷨
clear;
close all;
%Start Parallel Compute
core_number=25;            %��Ҫ���õĴ���������
parpool('local',core_number);
ProjectDir = pwd;
% ProjectDir='/home/lab421/MATLAB/projects/YanHaoqiuBgggr_constrastOnlyforCode';
% ProjectDir='C:\Users\admin\MATLAB\Projects\bgggr_constract';
% SysPathSeperator='\';
SysPathSeperator='/';
AutomsDir='atoms';
AutomsPath=[ProjectDir, SysPathSeperator, AutomsDir];
algorithm='ardgpr';
CsvoutputDir=[algorithm, '_output'];
CsvoutputPath=[ProjectDir, SysPathSeperator, CsvoutputDir];
diary([ProjectDir, SysPathSeperator, 'c2_ardgpr_log1.txt'])
%cd [ProjectDir, SysPathSeperator, RootDir]
AutomList=dir(AutomsPath); %���o1,c2,...
for j=1:length(AutomList)
    %����fileList�ṹ����ǰ���У���ɸ�������ļ��е�
    if strcmp(AutomList(j).name,'.')==1 || strcmp(AutomList(j).name,'..')==1 || ~AutomList(j).isdir
        continue;
    end
    autom=AutomList(j).name;
	if strcmp(autom,'c2') == 0
        disp(autom);
        continue 
	end
    DataSetsList=dir([AutomsPath, SysPathSeperator, autom, SysPathSeperator, '*.mat']);
    FitsTable=table;
    StatisticsTable=table;
    parfor i=1:length(DataSetsList)
        %����fileList�ṹ����ǰ����
        if strcmp(DataSetsList(i).name,'.')==1 || strcmp(DataSetsList(i).name,'..')==1
            continue;
        end
        DataSet=DataSetsList(i).name;
        DataSetNoExtn=strsplit(DataSet,'.');
        DataSetNoExtn=DataSetNoExtn{1};
        DataSetPath=[AutomsPath, SysPathSeperator, autom, SysPathSeperator, DataSetsList(i).name];
        disp(['DataSet is: ', DataSetPath])
        [ytest_fit, train_rmse, train_mse, test_rmse, test_mse, time]=ardgpr(DataSetPath);

        %��Ԥ��ֵ����table
        %eval(['FitsTable.', DataSetNoExtn, '=ytest_fit;']);
        ytest_fit_cell = num2cell(ytest_fit');
        FitsTable(i,:)=ytest_fit_cell;
           
        %��mse, time����table
        StatisticsTable(i, :)={train_rmse, train_mse, test_rmse,...
            test_mse, time, DataSetNoExtn};
    end
    delete(gcp('nocreate'))
%     %�ж���û�и�ԭ�ӵ��ļ���
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
     break
end
diary off