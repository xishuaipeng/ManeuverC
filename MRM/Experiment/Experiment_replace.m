
%%%%%%%%%%%%%%%%%%%%%Experiment 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear,clc;
close all;
addpath './OBDExtraction';
addpath './DataClass';
addpath './Feature'
addpath './Feature/VGG16'
addpath './Net'
videoName = { '028','112_07172017','118_07182017',...
    'ID001_T001','ID001_T002',...
    'ID001_T003','ID001_T004','ID001_T006','ID001_T007', ...
    'ID001_T008','ID001_T009','ID001_T010','ID001_T011','ID001_T012','ID001_T013',...
    'ID001_T014', 'ID001_T017', ...
    'ID002_T001','ID002_T002','ID002_T003','ID002_T004','ID002_T010', ...
    'ID002_T012','ID002_T015','ID002_T016','ID002_T017','ID002_T018', ...
    'ID002_T019','ID002_T020' ...
    'ID003_T001','ID003_T002','ID003_T003','ID003_T004','ID003_T010',...
    'ID005_T001','ID005_T002','ID005_T003','ID005_T004','ID005_T005','ID005_T007','ID005_T008',...
    'ID005_T009','ID005_T010','ID005_T012','ID005_T013','ID005_T014','ID005_T015'...
    'ID006_T001','ID006_T002','ID006_T003','ID006_T004','ID006_T006','ID006_T007',...
    'ID006_T009','ID006_T010',...
   }; 
% %%%
load('bigdata_time.mat')
% load('net.mat')
best_score = 0;
best_index = [1:size(data,1)];
all_index = [];
cur_data = data;
cur_label  = label;
for i =1:100
    cur_data = cur_data(best_index,:);
    cur_label = cur_label(best_index,:);
    [trainD,trainL,testD, testL,trainIndex, testIndex] = EventSplit(cur_data,cur_label,0.8);
    net = ManeuversNet(trainD, trainL,400,500);
    y = predict(net, testD );
    [ana_result,select_index] = eventAccuracy(y, cur_label(testIndex,:), 0.8);
    if ana_result.AllMean(3) >0.9
        break;
    end
    best_score
    length(testIndex)
    if ana_result.AllMean(3) > best_score
        best_score = ana_result.AllMean(3);
        all_index = [trainIndex; testIndex(select_index)];   
    end
%     y = predict(net, cur_data(select_index, :) );
%     [time_dis_ana,~] = eventAccuracy(y,cur_label(select_index,:), 0.8);
%     time_dis_ana
    best_index = [trainIndex; testIndex(select_index)];
end

%
    %'ID003_T001','ID003_T002','ID003_T003','ID003_T004','ID003_T010',...
    %'ID005_T001','ID005_T002','ID005_T003','ID005_T004','ID005_T005','ID005_T007','ID005_T008',...
    %'ID005_T009','ID005_T010','ID005_T012','ID005_T013','ID005_T014','ID005_T015'...
    %'ID006_T001','ID006_T002','ID006_T003','ID006_T004','ID006_T006','ID006_T007',...
    %'ID006_T009','ID006_T010',...
%%%
%,can not read: 'ID001_T015', 'D003_T014', 'ID005_T011','ID006_T005',,'ID006_T008''ID002_T013',
      %unlabeled: 'ID005_T012','ID005_T013','ID005_T014','ID005_T015'
      %night '106_07142017''023'
% data = [];
% label = [];
% tic;
% %        'event_checklist_path',checkFile ,...
% for i =1:length(videoName)
%     sprintf('%s is processing!',videoName{i})
%     input_path = 'D:\xishuaip\TRI\Project\Dataset';
%     out_path = 'D:\xishuaip\TRI\Project\process';
%     checkFile = sprintf('%s/%s/%s.list',input_path,videoName{i}, videoName{i});
%     Mdata = Maneuverdata('data_id', videoName{i},...
%         'input_path',input_path,...
%         'log_field', {'time','speed','GPS_long','GPS_lat','GPS_heading','distance'},...
%         'out_dir',out_path,...
%          'start_padding',10,...
%          'last_padding',5,...
%          'event_checklist_path',checkFile ,...
%          'gene_feature', {'Curvature','VGGObject','Heading','Speed'},...%,'VGGScene'
%          'load_feature', {'Curvature','VGGObject','Heading','Speed'},...%
%          'samplingStep',0.1,...  
%          'samplingProperty','time',...    
%          'seq_window',3,...
%          'seq_step', 0.3...
%         );
%     Mdata.WriteEvent();
%     [seqX,seqY] = Mdata.EventSequence(i);
%     data = [data;seqX];
%     label = [label;seqY];
% end
% toc;
% %%%%%%%%
% [trainD,trainL,testD, testL,trainIndex, testIndex] = EventSplit(data,label,0.8);
% % fprintf('Ite:%d \n', n);
% net = ManeuversNet(trainD, trainL,400,500);
% %
% display('training');
% y = predict(net, trainD );
% [time_dis_ana,TP]  = eventAccuracy(y,label(trainIndex,:), 0.8);
% time_dis_ana
% 
% % only for event accuracy
% display('testing');
% y = predict(net, testD);
% [time_dis_ana,TP] = eventAccuracy(y,label(testIndex,:), 0.8);
% time_dis_ana

%


















