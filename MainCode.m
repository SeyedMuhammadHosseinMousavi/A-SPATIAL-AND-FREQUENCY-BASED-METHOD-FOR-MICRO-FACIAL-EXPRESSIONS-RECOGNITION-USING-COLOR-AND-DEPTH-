%% 
% This code implements the following paper:
% A SPATIAL AND FREQUENCY BASED METHOD FOR MICRO FACIAL EXPRESSIONS 
% RECOGNITION USING COLOR AND DEPTH IMAGESA SPATIAL AND FREQUENCY BASED 
% METHOD FOR MICRO FACIAL EXPRESSIONS RECOGNITION USING COLOR AND DEPTH
% IMAGES, Journal of Software Engineering & Intelligent Systems 6 (1), 17,
% (2021).

% -In order to get the samples, it is required to send me a letter from your
% supervisor that it is going to be used just for your scientific experiment 
% and responsibility of any other usage is with your supervisor.

% My Email:    mosavi.a.i.buali@gmail.com

% Mousavi, Seyed Muhammad Hossein, and S. Younes Mirinezhad.
% "Iranian Kinect face database (IKFDB): a color-depth based face database
% collected by Kinect v. 2 sensor." SN Applied Sciences 3.1 (2021): 1-17.

clear;clc;

%% Data Reading and Pre-Processing
path='IKFDB';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
    disp(['Loading image No :   ' num2str(i) ]);
end;
% Resize Images
for i = 1 : filesnumber(1,1)
resized{i}=imresize(images{i}, [256 128]); 
    disp(['Image Resized :   ' num2str(i) ]);
end;
% Color to Gray Conversion
for i = 1 : filesnumber(1,1)
gray{i}=rgb2gray(resized{i}); 
    disp(['To Gray :   ' num2str(i) ]);
end;
% Contrast Adjustment
for i = 1 : filesnumber(1,1)
adjusted{i}=imadjust(gray{i}); 
    disp(['Image Adjust :   ' num2str(i) ]);
end;
%% Feature Extraction
% Extract SURF Features (just color)
imset = imageSet('IKFDB SURF','recursive'); 
% Create a bag-of-features from the image database
bag = bagOfFeatures(imset,'VocabularySize',20,'PointSelection','Detector');
% Encode the images as new features
surf = encode(bag,imset);

% Extract LPQ Features 
% More value for winsize, better result
winsize=29;
for i = 1 : filesnumber(1,1)
tmp{i}=lpq(adjusted{i},winsize);
disp(['Extract LPQ :   ' num2str(i) ]);end;
for i = 1 : filesnumber(1,1)lpq(i,:)=tmp{i};end;

% Extract LBP Features  
for i = 1 : filesnumber(1,1)
    % The less cell size the more accuracy 
lbp{i} = extractLBPFeatures(adjusted{i},'CellSize',[90 90]);
    disp(['Extract LBP :   ' num2str(i) ]);
end;
clear lbpfeature;
for i = 1 : filesnumber(1,1)
    lbpfeature(i,:)=lbp{i};
    disp(['LBP To Matrix :   ' num2str(i) ]);
end;

% Extract HOG Features (just color)
for i = 1 : filesnumber(1,1)
    % The less cell size the more accuracy 
hog{i} = extractHOGFeatures(adjusted{i},'CellSize',[45 45]);
    disp(['Extract HOG :   ' num2str(i) ]);
end;
for i = 1 : filesnumber(1,1)
    hogfeature(i,:)=hog{i};
    disp(['HOG To Matrix :   ' num2str(i) ]);
end;
LBP=lbpfeature;HOG=hogfeature;LPQ=lpq;SURF=surf;

% Combining Feature Matrixes
FinalReady=[LBP HOG LPQ SURF];

% % lasso feature selection
% disp(['Working On Lasso Feature Selection (Please Wait) ...']);
% % Labeling for lasso
% clear lasso;clear B;clear Stats;clear ds;
% label(1:200,1)=1;
% label(201:400,1)=2;
% label(401:600,1)=3;
% label(601:800,1)=4;
% label(801:1000,1)=5;
% % clear lasso;
% [B Stats] = lasso(FinalReady,label, 'CV', 5);
% disp(B(:,1:5))
% disp(Stats)
% lassoPlot(B, Stats, 'PlotType', 'CV')
% ds.Lasso = B(:,Stats.IndexMinMSE);
% disp(ds)
% sizemfcc=size(FinalReady);
% temp=1;       
% for i=1:sizemfcc(1,2)
% if ds.Lasso(i)~=0
% lasso(:,temp)=FinalReady(:,i);
% temp=temp+1;end;end;FinalReady=lasso;

% Labels
sizefinal=size(FinalReady);
sizefinal=sizefinal(1,2);
FinalReady(1:200,sizefinal+1)=1;
FinalReady(201:400,sizefinal+1)=2;
FinalReady(401:600,sizefinal+1)=3;
FinalReady(601:800,sizefinal+1)=4;
FinalReady(801:1000,sizefinal+1)=5;

%%
% Classification
% KNN 
lblknn=FinalReady(:,end);
dataknn=FinalReady(:,1:end-1);

tknn = templateKNN('NumNeighbors',5,'Standardize',1);
Mdl = fitcensemble(dataknn,lblknn,'Method','Subspace','Learners',tknn);
rng(1); % For reproducibility
knndat = crossval(Mdl);
classError = kfoldLoss(knndat)
Lknn = resubLoss(Mdl,'LossFun','classiferror'); 
KNNAccuracy = 1 - kfoldLoss(knndat, 'LossFun', 'ClassifError');
% Predict the labels of the training data.
predictedknn = resubPredict(Mdl);
% Plot Confusion Matrix
figure
cmknn = confusionchart(lblknn,predictedknn);
cmknn.Title = 'KNN';
cmknn.RowSummary = 'row-normalized';
cmknn.ColumnSummary = 'column-normalized';
% Precision, Recall and ROC
[~,scoreknn] = resubPredict(Mdl);
diffscoreknn = scoreknn(:,2) - max(scoreknn(:,1),scoreknn(:,3));
[Xknn,Yknn,T,~,OPTROCPTknn,suby,subnames] = perfcurve(lblknn,diffscoreknn,1);
%
figure;
plot(Xknn,Yknn)
hold on
plot(OPTROCPTknn(1),OPTROCPTknn(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for KNN')
hold off
%
knnsss=size(Xknn);knnsss=knnsss(1,1);
mx=min(Xknn(Xknn>0));
Preknn=max(Xknn)-mx;
my=min(Yknn(Yknn>0));
Recknn=max(Yknn)-my;
disp(['K-NN Precision :   ' num2str(Preknn) ]);
disp(['K-NN Recall :   ' num2str(Recknn) ]);

% SVM
tsvm = templateSVM('KernelFunction','polynomial');
svmclass = fitcecoc(dataknn,lblknn,'Learners',tsvm);
svmerror = resubLoss(svmclass);
CVMdl = crossval(svmclass);
genError = kfoldLoss(CVMdl);
% Compute validation accuracy
SVMAccuracy = 1 - kfoldLoss(CVMdl, 'LossFun', 'ClassifError');
% Predict the labels of the training data.
predictedsvm = resubPredict(svmclass);
% Plot Confusion Matrix
figure
cmsvm = confusionchart(lblknn,predictedsvm);
cmsvm.Title = 'SVM';
cmsvm.RowSummary = 'row-normalized';
cmsvm.ColumnSummary = 'column-normalized';
% Precision, Recall and ROC
[~,scoresvm] = resubPredict(svmclass);
diffscoresvm = scoresvm(:,2) - max(scoresvm(:,1),scoresvm(:,3));
[Xsvm,Ysvm,T,~,OPTROCPTsvm,suby,subnames] = perfcurve(lblknn,diffscoresvm,1);
%
figure;
plot(Xsvm,Ysvm)
hold on
plot(OPTROCPTsvm(1),OPTROCPTsvm(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for SVM')
hold off
%
svmsss=size(Xsvm);
svmsss=svmsss(1,1);
mx=min(Xsvm(Xsvm>0));
my=min(Ysvm(Ysvm>0));
Presvm=max(Xsvm)-mx;
Recsvm=max(Ysvm)-my;
disp(['SVM Precision :   ' num2str(Presvm) ]);
disp(['SVM Recall :   ' num2str(Recsvm) ]);

% LDA
MdlLinear = fitcdiscr(dataknn,lblknn);
Lin = resubLoss(MdlLinear);
ldadat = crossval(MdlLinear);
LDAAccuracy = 1 - kfoldLoss(ldadat, 'LossFun', 'ClassifError');
% Predict the labels of the training data.
predictedlda = resubPredict(MdlLinear);
% Plot Confusion Matrix
figure
cmlda = confusionchart(lblknn,predictedlda);
cmlda.Title = 'LDA';
cmlda.RowSummary = 'row-normalized';
cmlda.ColumnSummary = 'column-normalized';
% Precision, Recall and ROC
[~,scorelda] = resubPredict(MdlLinear);
diffscorelda = scorelda(:,2) - max(scorelda(:,1),scorelda(:,3));
[Xlda,Ylda,T,~,OPTROCPTlda,suby,subnames] = perfcurve(lblknn,diffscorelda,3);
%
figure;
plot(Xlda,Ylda)
hold on
plot(OPTROCPTlda(1),OPTROCPTlda(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for LDA')
hold off
%
ldasss=size(Xlda);
ldasss=ldasss(1,1);
mx=min(Xlda(Xlda>0));
my=min(Ylda(Ylda>0));
Prelda=max(Xlda)-mx;
Reclda=max(Ylda)-my;
disp(['LDA Precision :   ' num2str(Prelda) ]);
disp(['LDA Recall :   ' num2str(Reclda) ]);
%% Shallow Neural Network
% Neural Network
network=FinalReady(:,1:end-1);
netlbl=FinalReady(:,end);
sizenet=size(network);
sizenet=sizenet(1,1);
for i=1 : sizenet
            if netlbl(i) == 1
               netlbl2(i,1)=1;
        elseif netlbl(i) == 2
               netlbl2(i,2)=1;
        elseif netlbl(i) == 3
               netlbl2(i,3)=1; 
        elseif netlbl(i) == 4
               netlbl2(i,4)=1;
        elseif netlbl(i) == 5
               netlbl2(i,5)=1; 
        end
end
% Changing data shape from rows to columns
network=network'; 
% Changing data shape from rows to columns
netlbl2=netlbl2'; 
% Defining input and target variables
inputs = network;
targets = netlbl2;
% Create a Pattern Recognition Network
hiddenLayerSize = 100;
net = patternnet(hiddenLayerSize);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the Network
% Polak-Ribiére Conjugate Gradient
net = feedforwardnet(10, 'traincgp');
%
[net,tr] = train(net,inputs,targets);
% Test the Network
outputs = net(inputs);
%
errors = gsubtract(targets,outputs);
%
performance = perform(net,targets,outputs);
% Polak-Ribiére Conjugate Gradient
figure, plottrainstate(tr)
% Plot Confusion Matrixes
figure, plotconfusion(targets,outputs);
title('Polak-Ribiére Conjugate Gradient');
% Res
disp(['K-NN Classification Accuracy :   ' num2str(KNNAccuracy*100) ]);
disp(['SVM Classification Accuracy :   ' num2str(SVMAccuracy*100) ]);
disp(['LDA Classification Accuracy :   ' num2str(LDAAccuracy*100) ]);
disp(['Shallow NN Classification Accuracy :   ' num2str(100-performance) ]);

%% Deep Neural Network
% CNN Facial Expressions
deepDatasetPath = fullfile('IKFDB CNN');
imds = imageDatastore(deepDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% Number of training (less than number of each class)
numTrainFiles = 400;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
layers = [
    % Input image size for instance: 512 512 3
    imageInputLayer([128 128 1])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    % Number of classes
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',9, ...
    'Verbose',false, ...
    'Plots','training-progress');
netmacro = trainNetwork(imdsTrain,layers,options);
YPred = classify(netmacro,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation) *100;
disp(['CNN Macro Recognition Accuracy Is =   ' num2str(accuracy) ]);

%% Deep Neural Network
% CNN Micro Facial Expressions
deepDatasetPath = fullfile('Micro');
imdsmicro = imageDatastore(deepDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% Number of training (less than number of each class)
numTrainFiles = 92;
[imdsTrain,imdsValidation] = splitEachLabel(imdsmicro,numTrainFiles,'randomize');
layers = [
    % Input image size for instance: 512 512 3
    imageInputLayer([128 128 3])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    % Number of classes
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',50, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',9, ...
    'Verbose',false, ...
    'Plots','training-progress');
netmicro = trainNetwork(imdsTrain,layers,options);
YPred = classify(netmicro,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation) *100;
disp(['CNN Micro Recognition Accuracy Is =   ' num2str(accuracy) ]);

