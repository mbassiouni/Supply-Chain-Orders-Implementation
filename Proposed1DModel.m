load 'DataOrders.mat';
TrainData = normalize(ExcelSmallDataSmaller(1:108312,1:13));
ValidationData = normalize(ExcelSmallDataSmallerTest(1:36103,1:13));
TrainData = TrainData';
for i = 1 : 108312
   XTrain{i} = TrainData(:,i); 
end
XTrain = XTrain';
ValidationData = ValidationData';
for i = 1 : 36103
   XValid{i} = ValidationData(:,i); 
end
XValid = XValid';
FoldTrainLabels = ConvertLabelsNumber_To_Categorial(TrainClasses(1:108312,1));
FoldTrainLabels = FoldTrainLabels';
FoldValidLabels = ConvertLabelsNumber_To_Categorial(TestClasses(1:36103,1));
FoldValidLabels = FoldValidLabels';
numFeatures = 13;
numClasses = 2;
x = 0;
filterSize = 3;
numFilters = 32;
layers = [ ...
    sequenceInputLayer(numFeatures)
    convolution1dLayer(filterSize,numFilters,Padding="causal")
    reluLayer
    layerNormalizationLayer
    convolution1dLayer(filterSize,2*numFilters,Padding="causal")
    reluLayer
    layerNormalizationLayer
    convolution1dLayer(filterSize,3*numFilters,Padding="causal")
    reluLayer
    layerNormalizationLayer
    globalAveragePooling1dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
analyzeNetwork(layers);
miniBatchSize = 32;
options = trainingOptions("adam", ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=15, ...
    SequencePaddingDirection="left", ...
    ValidationData={XValid,FoldValidLabels}, ...
    ValidationFrequency=3384, ...
    Plots="training-progress", ...
    Verbose=1);
TrainData = TrainData';
net = trainNetwork(XTrain,FoldTrainLabels,layers,options);
x = 0;
