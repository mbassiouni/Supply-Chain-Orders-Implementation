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
numFilters = 32;
filterSize = 3;
dropoutFactor = 0.1;
numBlocks = 2;
layer = sequenceInputLayer(numFeatures,Normalization="rescale-symmetric",Name="input");
lgraph = layerGraph(layer);
outputName = layer.Name;
for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    
    layers = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        layerNormalizationLayer
        dropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
        layerNormalizationLayer
        reluLayer
        dropoutLayer(dropoutFactor)
        additionLayer(2,Name="add_"+i)];

    % Add and connect layers.
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection.
    if i == 1
        % Include convolution in first skip connection.
        layer = convolution1dLayer(1,numFilters,Name="convSkip");

        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,"convSkip");
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
    else
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
    end
    
    % Update layer output name.
    outputName = "add_" + i;
end
filterSize = 3;
numFilters = 32;
layers = [ ...
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


lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,'add_2','conv1d_2');
analyzeNetwork(lgraph);


miniBatchSize = 32;
options = trainingOptions("rmsprop", ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=15, ...
    SequencePaddingDirection="left", ...
    ValidationData={XValid,FoldValidLabels}, ...
    ValidationFrequency=500, ...
    Plots="training-progress", ...
    Verbose=1);
TrainData = TrainData';
net = trainNetwork(XTrain,FoldTrainLabels,lgraph,options);
