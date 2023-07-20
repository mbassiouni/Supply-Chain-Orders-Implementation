function [XTrain,FoldTrainLabels,lgraph,options] =ProposedTCN1DSPCNNModel(params)
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
numFilters = 64;
filterSize = 5;
dropoutFactor = 0.005;
numBlocks = params.B1;
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
nolayers = params.B2;
for i = 1 : nolayers
layers = [ ...
    convolution1dLayer(filterSize,numFilters,Padding="causal",Name="conv11_"+i)
    reluLayer
    layerNormalizationLayer(Name="layernorm__"+i)
    ];
   if i == 1
   lgraph = addLayers(lgraph,layers);
   lgraph = connectLayers(lgraph,"add_"+numBlocks,"conv11_"+i);
   else 
   lgraph = addLayers(lgraph,layers);
   lgraph = connectLayers(lgraph,"layernorm__"+(i-1),"conv11_"+i);
   end
end
% Finallayers = [...
%     globalAveragePooling1dLayer(Name ="ga")];
% lgraph = addLayers(lgraph,Finallayers);
% lgraph = connectLayers(lgraph,"layernorm__"+(nolayers),"ga");

numB = params.B3;
numFilters = 8;
for j = 1:numB
    N = 3;
    block = [
        convolution1dLayer(N,numFilters,Name="conv"+j,Padding="same")
        batchNormalizationLayer(Name="bn"+j)
        reluLayer(Name="relu"+j)
        dropoutLayer(0.2,Name="drop"+j)
        globalMaxPooling1dLayer(Name="max"+j)];
    numFilters = numFilters * 2;
    if j == 1
    lgraph = addLayers(lgraph,block);
    lgraph = connectLayers(lgraph,"layernorm__"+(nolayers),"conv"+j);
   
    else
    lgraph = addLayers(lgraph,block);
    lgraph = connectLayers(lgraph,"layernorm__"+nolayers,"conv"+j);
    end
end
layers = [
    concatenationLayer(1,numB,Name="cat")];
lgraph = addLayers(lgraph,layers);
for j = 1:numB
    lgraph = connectLayers(lgraph,"max"+j,"cat/in"+j);
end
Finallayers = [...
    fullyConnectedLayer(numClasses,Name="fc")
    softmaxLayer
    classificationLayer];
lgraph = addLayers(lgraph,Finallayers);
lgraph = connectLayers(lgraph,"cat","fc");
miniBatchSize = 32;
options = trainingOptions("adam", ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=25, ...
    ValidationData={XValid,FoldValidLabels}, ...
    ValidationFrequency=4074, ...
    LearnRateSchedule= 'piecewise',...
    LearnRateDropPeriod= params.LP,...
    LearnRateDropFactor= params.LF,...
    InitialLearnRate = params.myInitialLearnRate, ...
    L2Regularization = params.L2Regularization, ...
    Verbose=0);
   % 'LearnRateDropPeriod',params.LP,...
   % 'LearnRateDropFactor',params.LF,...  
 %   'InitialLearnRate', params.myInitialLearnRate, ...
 %   'L2Regularization',params.L2Regularization, ...   
end
