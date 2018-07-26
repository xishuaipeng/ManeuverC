function net = ManeuversNet(X, Y,outputSize,maxEpochs)
%--------------------------------------------------------------------------
% MANEUVERSNET: this function is the main function to train the network
%
%   Function Signature:
%         net = ManeuversNet(X, Y,outputSize,maxEpochs)
%         @input:
%             X             : 
%             Y             :
%             outputSize    :
%             maxEpochs     :
%
%         @output:
%             net           :
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% @author: Xishuai Peng
% @email: xishuaip@umich.edu
% @date: May.30.2018
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%--------------------------------------------------------------------------
[num_feature, temp] = size(X{1});
inputSize = num_feature;
sprintf('Feature Size: %f', inputSize)
outputMode = 'sequence';
numClasses = 5;
%,'InputWeightsL2Factor',0.01
layers = [ ...
    sequenceInputLayer(inputSize)
    %bilstmLayer()
    lstmLayer(outputSize,'OutputMode',outputMode,'InputWeightsL2Factor',0.2);%
    lstmLayer(outputSize/2,'OutputMode',outputMode,'InputWeightsL2Factor',0.2)
    lstmLayer(outputSize/4,'OutputMode','last','InputWeightsL2Factor',0.3)
    dropoutLayer(0.1)
    fullyConnectedLayer(numClasses) 
    softmaxLayer
    classificationLayer];
miniBatchSize = 50;
%     'LearnRateDropFactor',0.1,...
%     'LearnRateDropPeriod',300,...
%     'LearnRateSchedule' ,'piecewise',...
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01,...
    'MaxEpochs',maxEpochs, ...
    'ValidationFrequency',100,...
    'MiniBatchSize',miniBatchSize,...
    'Shuffle', 'every-epoch',...
    'Plots','training-progress');
%    %'CheckpointPath','./model',...
%every-epoch
net = trainNetwork(X,Y,layers,options);

end