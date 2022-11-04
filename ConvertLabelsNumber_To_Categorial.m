function [ResClasses] = ConvertLabelsNumber_To_Categorial (Classes)
[r,c] = size(Classes);
for i = 1 : r
   ResClasses(i) = categorical(Classes(i)); 
end
end