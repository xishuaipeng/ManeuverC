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
%%%
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
data = [];
label = [];
tic;
%        'event_checklist_path',checkFile ,...
for i =1:length(videoName)
    sprintf('%s is processing!',videoName{i})
    input_path = 'D:\xishuaip\TRI\Project\Dataset';
    out_path = 'D:\xishuaip\TRI\Project\process_time';
    checkFile = sprintf('%s/%s/%s.list',input_path,videoName{i}, videoName{i});
    disp( remove_mat(fullfile(out_path,videoName{i}),{'seqX_5.000000_1.000000_10.000000_5.000000.mat';'seqY_5.000000_1.000000_10.000000_5.000000.mat'} ));
    Mdata = Maneuverdata('data_id', videoName{i},...
        'input_path',input_path,...
        'log_field', {'time','speed','GPS_long','GPS_lat','GPS_heading','distance'},...
        'out_dir',out_path,...
         'start_padding',10,...
         'last_padding',5,...
         'event_checklist_path',checkFile ,...
         'gene_feature', {'Curvature','VGGObject','Heading','Speed'},...%,'VGGScene'
         'load_feature', {'Curvature','VGGObject','Heading','Speed'},...%
         'samplingStep',0.1,...  
         'samplingProperty','time',...    
         'seq_window',3,...
         'seq_step', 0.3 ...
        );
    %Mdata.WriteEvent();
    [seqX,seqY] = Mdata.EventSequence(i);
    data = [data;seqX];
    label = [label;seqY];
end
toc;
%%%%%%%%

[trainD,trainL,testD, testL,trainIndex, testIndex] = EventSplit(data,label,0.8);
% fprintf('Ite:%d \n', n);
net = ManeuversNet(trainD, trainL,400,500);
%
display('training');
y = predict(net, trainD );
[time_dis_ana,TP] = eventAccuracy(y,label(trainIndex,:), 0.8);
time_dis_ana
% only for event accuracy
display('testing');
% y_y = classify(net, testD );
% disp(sum(label(testIndex,:).eventType == double(y_y))/ length(y_y))
% %ture_index = label(testIndex,:).eventType == double(y_y);
% % y = predict(net, testD );
% % [time_dis_ana,TP] = eventAccuracy(y,label(testIndex(ture_index),:), 0.8);
% ture_index = find(sum(TP,2)==1);
y = predict(net, testD);
[time_dis_ana,TP] = eventAccuracy(y,label(testIndex,:), 0.8);
time_dis_ana
%


















