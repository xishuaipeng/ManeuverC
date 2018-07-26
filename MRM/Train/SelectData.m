function [trainD,trainL,testD, testL,testIdx] = selectData(trainData, Label, ratio, varargin)
%--------------------------------------------------------------------------
% SELECTDATA: this function is to split the processed dataset to train
% datasest and test destset
%
%   Function Signature:
%         [trainD,trainL,testD, testL,testIdx] = Balancesplit(trainData, Label, ratio)
%         [trainD,trainL,testD, testL,testIdx] = Balancesplit(trainData, Label, ratio, k)
%         @input:
%             trainData     : 
%             Label         :
%             ratio         :
%             k             :
%
%         @output:
%             trainD        :
%             trainL        :
%             testD         :
%             testL         :
%             testIdx       :
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% @author: Xishuai Peng
% @email: xishuaip@umich.edu
% @date: May.30.2018
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%--------------------------------------------------------------------------
minL = min(Label);
maxL = max(Label);
for i = minL: maxL
    idx(i+1) = {find(Label==i)};
end
if nargin == 4
    k = varargin{4};
else 
    k = 4;
end
negNum = round(ratio*k*mean([length(idx{2}),length(idx{3}), ...
    length(idx{4}),length(idx{5})]));
trainIdx = [];
for i = minL:maxL
    if i == minL
        if negNum<length(idx{i+1})
            trainNum = negNum;
        else
            trainNum = round(ratio*length(idx{i+1}));
        end
    else
        trainNum = round(ratio*length(idx{i+1}));
    end
    idx_ran = randperm(length(idx{i+1}));
    trainIdx = [trainIdx; idx{i+1}(idx_ran(1:trainNum))]; 
end
testIdx = 1:length(Label);
testIdx(trainIdx) = [];
trainIdx = trainIdx(randperm(length(trainIdx)));
testIdx = testIdx(randperm(length(testIdx)));
trainD = trainData(trainIdx);
trainL =  categorical(Label(trainIdx));
testD =  trainData(testIdx);
testL =  categorical(Label(testIdx));
end

