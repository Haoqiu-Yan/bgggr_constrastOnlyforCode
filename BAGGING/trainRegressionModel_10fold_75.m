function [trainedModel, validationRMSE] = trainRegressionModel_10fold_75(trainingData,leaf,learning)
% [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% ���ؾ���ѵ���Ļع�ģ�ͼ��� RMSE�����´������´����� Regression Learner App ��ѵ����
% ģ�͡�������ʹ�ø����ɵĴ�������������Զ�ѵ��ͬһģ�ͣ���ͨ�����˽�����Գ��򻯷�ʽѵ��ģ
% �͡�
%
%  ����:
%      trainingData: һ���������������������뵼�� App �е���ͬ�ľ���
%
%  ���:
%      trainedModel: һ������ѵ���Ļع�ģ�͵Ľṹ�塣�ýṹ���о��и��ֹ�����ѵ��ģ�͵�
%       ��Ϣ���ֶΡ�
%
%      trainedModel.predictFcn: һ���������ݽ���Ԥ��ĺ�����
%
%      validationRMSE: һ������ RMSE ��˫����ֵ���� App �У�"��ʷ��¼" �б���ʾÿ��
%       ģ�͵� RMSE��
%
% ʹ�øô��������������ѵ��ģ�͡�Ҫ����ѵ��ģ�ͣ���ʹ��ԭʼ���ݻ���������Ϊ�������
% trainingData �������е��øú�����
%
% ���磬Ҫ����ѵ������ԭʼ���ݼ� T ѵ���Ļع�ģ�ͣ�������:
%   [trainedModel, validationRMSE] = trainRegressionModel(T)
%
% Ҫʹ�÷��ص� "trainedModel" �������� T2 ����Ԥ�⣬��ʹ��
%   yfit = trainedModel.predictFcn(T2)
%
% T2 �����ǽ���������ѵ����Ԥ������еľ����й���ϸ��Ϣ��������:
%   trainedModel.HowToPredict

% �� MATLAB �� 2019-12-29 01:14:16 �Զ�����


% ��ȡԤ���������Ӧ
% ���´��뽫���ݴ���Ϊ���ʵ���״��ѵ��ģ�͡�
%
% ������ת��Ϊ��
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58', 'column_59', 'column_60', 'column_61', 'column_62', 'column_63', 'column_64', 'column_65', 'column_66', 'column_67', 'column_68', 'column_69', 'column_70', 'column_71', 'column_72', 'column_73', 'column_74', 'column_75', 'column_76'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58', 'column_59', 'column_60', 'column_61', 'column_62', 'column_63', 'column_64', 'column_65', 'column_66', 'column_67', 'column_68', 'column_69', 'column_70', 'column_71', 'column_72', 'column_73', 'column_74', 'column_75'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_76;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% ѵ���ع�ģ��
% ���´���ָ������ģ��ѡ�ѵ��ģ�͡�
template = templateTree(...
    'MinLeafSize', leaf);
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', learning, ...
    'Learners', template);

% ʹ��Ԥ�⺯����������ṹ��
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% �����ṹ���������ֶ�
trainedModel.RegressionEnsemble = regressionEnsemble;
trainedModel.About = '�˽ṹ���Ǵ� Regression Learner R2019b ������ѵ��ģ�͡�';
trainedModel.HowToPredict = sprintf('Ҫ����Ԥ������о��� X ����Ԥ�⣬��ʹ��: \n yfit = c.predictFcn(X) \n�� ''c'' �滻Ϊ��Ϊ�˽ṹ��ı��������ƣ����� ''trainedModel''��\n \nX ����������� 75 ���У���Ϊ��ģ����ʹ�� 75 ��Ԥ���������ѵ���ġ�\nX �����������ѵ�����ݾ�����ȫ��ͬ��˳��͸�ʽ��\nԤ������С���Ҫ������Ӧ�л�δ���� App ���κ��С�\n \n�й���ϸ��Ϣ������� <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>��');

% ��ȡԤ���������Ӧ
% ���´��뽫���ݴ���Ϊ���ʵ���״��ѵ��ģ�͡�
%
% ������ת��Ϊ��
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58', 'column_59', 'column_60', 'column_61', 'column_62', 'column_63', 'column_64', 'column_65', 'column_66', 'column_67', 'column_68', 'column_69', 'column_70', 'column_71', 'column_72', 'column_73', 'column_74', 'column_75', 'column_76'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58', 'column_59', 'column_60', 'column_61', 'column_62', 'column_63', 'column_64', 'column_65', 'column_66', 'column_67', 'column_68', 'column_69', 'column_70', 'column_71', 'column_72', 'column_73', 'column_74', 'column_75'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_76;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% ִ�н�����֤
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', 10);

% ������֤Ԥ��
validationPredictions = kfoldPredict(partitionedModel);

% ������֤ RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));