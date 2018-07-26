function [trainD,trainL,testD, testL,trainIndex, testIndex] = EventSplit(data,label,radio)
%--------------------------------------------------------------------------
% BALANCESPLIT: this function is to split the processed dataset to train
% datasest and test destset based on each event
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
[id, ia,ic] = unique(label.sequenceIndex,'stable');
event_num = length(id) ;
label_type = label.sequenceLabel;
id_event_type = label_type(ia);
min_event_type = min(label_type);
max_event_type = max(label_type);
train_id = [];
test_id = [];
train_index = [];
test_index = [];
for i = min_event_type: max_event_type
    current_type_id = find(id_event_type==i);
    current_type_num = length(current_type_id);
    train_num = floor(current_type_num * radio);
    index = randperm(current_type_num );
    train_index_cur = index(1:train_num);
    test_index_cur = index(train_num+1:end);
    train_id = [train_id; current_type_id(train_index_cur)];
    test_id = [test_id; current_type_id(test_index_cur)];  
end

 for i = 1: length(train_id)
     id_cur =id( train_id(i));
     train_index_cur = find(label.sequenceIndex == id_cur) ;
     train_index = [train_index;train_index_cur ] ;  
 end

for i = 1: length(test_id)
    id_cur = id(test_id(i));
    test_index_cur = find(label.sequenceIndex == id_cur) ;
    test_index = [test_index; test_index_cur ]   ;
end

trainD = data(train_index);
trainL = categorical(label_type(train_index));
testD = data(test_index);
testL = categorical(label_type(test_index));
testIndex = test_index;
trainIndex = train_index;
end