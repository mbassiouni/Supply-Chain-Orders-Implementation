function [featureadded] = ConvertTypeData ()
load 'Data.mat';
for i = 1 : 180519
    featurecell(i) = ExcelSmallDataTest(i,1);
    feature(i) = featurecell{i};
    if strcmp(feature(i),"DEBIT")
        val(i) = 11;
    elseif strcmp(feature(i),"TRANSFER")
        val (i) = 12;
    elseif strcmp(feature(i),"CASH")
        val(i) = 13;
    elseif strcmp(feature(i),"PAYMENT")
        val(i) = 14;
    end
end
featureadded = val;
end