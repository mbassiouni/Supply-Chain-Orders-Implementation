function [featureadded] = ConvertOrderStatus()
load 'Data2.mat';
for i = 1 : 180519
    featurecell(i) = DataCoSupplyChainDataset(i,:);
    feature(i) = featurecell(i);
    if strcmp(feature(i),"COMPLETE")
        val(i) = 1000;
    elseif strcmp(feature(i),"PENDING")
        val (i) = 1001;
    elseif strcmp(feature(i),"CLOSED")
        val(i) = 1002;
    elseif strcmp(feature(i),"PENDING_PAYMENT")
        val(i) = 1003;
    elseif strcmp(feature(i),"PROCESSING")
        val(i) = 1004;
    elseif strcmp(feature(i),"CANCELED")
        val(i) = 1005;
    elseif strcmp(feature(i),"ON_HOLD")
        val(i) = 1006;
    elseif strcmp(feature(i),"SUSPECTED_FRAUD")
        val(i) = 1007;
    end
end
featureadded = val;
end