function zc
%GENERATEACCURACY generate precesion and recall
%   testAna = generateAccuracy(testL, z)
% testL is label result
% z is detection result
% if testL or z is categorical, pls double(testL)-1 first.
% accuracy is with STR
testAna = table();
for i = 1: 5
    if i == 1
        testAna.reIdx(i) = {find(testL~=0)};
        testAna.prIdx(i) = {find(z~=0)};
    else
        testAna(i,{'reIdx', 'prIdx'}) = [num2cell({find(testL==i-1)}) , ...
            num2cell({find(z==i-1)})];
    end
end
rowNames = {'all', 'ltr', 'rtr', 'llc', 'rlc'};
testAna.Properties.RowNames = rowNames;
testAna(:, {'Recall', 'Re_Num', 'Precision', 'Pr_Num'}) = ...
        array2table(zeros(length(rowNames),4));
for i = 1 : length(rowNames)
    index_re = cell2mat(table2array(testAna(rowNames{i}, 'reIdx')));
    index_pr = cell2mat(table2array(testAna(rowNames{i}, 'prIdx')));
    recall = {sum(z(index_re) == testL(index_re))/length(index_re)};
    re_num = {length(index_re)};
    precision = {sum(z(index_pr) == testL(index_pr))/length(index_pr)};
    pr_num = {length(index_pr)};
    testAna(i, {'Recall', 'Re_Num', 'Precision', 'Pr_Num'}) = ...
        [recall, re_num, precision, pr_num];
    
end
accuracy =  sum(z == testL)/numel(testL);
end

