function [trainD,trainL,testD, testL,testIndex] = SplitData(data,label,radio)
%--------------------------------------------------------------------------
% SPLITDATA: this function is to split the processed dataset to train
% datasest and test destset based on all the events
%
%   Function Signature:
%         [trainD,trainL,testD, testL,testIndex] = SplitData(data,label,radio)
%         @input:
%             data     : 
%             label         :
%             ratio         :
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
nCase = size(label,1);
randomIndex = randperm(nCase);
nTrain = nCase * radio;
trainIndex = randomIndex(1:nTrain);
testIndex = randomIndex(nTrain:end);
trainD = data(trainIndex);
trainL =  categorical(label(trainIndex));
testD =  data(testIndex);
testL =  categorical(label(testIndex));






end

