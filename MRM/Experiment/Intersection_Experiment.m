%%%%%%%%%%%%%%%%%%%%%Experiment 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear,clc;
close all;
addpath './OBDExtraction';
addpath './DataClass';
addpath './Feature'
addpath './Feature/VGG16'
addpath './Net'
videoName = {'ID008_T004','ID008_T005','ID008_T006','ID008_T007','ID008_T008','ID008_T009','ID008_T010','ID008_T011','ID008_T012','ID008_T013','ID008_T014','ID008_T015'};
%,'I'ID005_T001','ID005_T002','ID005_T003','ID005_T004','ID005_T005','ID005_T007','ID005_T008','ID005_T012','ID005_T013','ID005_T014',...%'ID005_T011','ID005_T001','ID005_T002','ID005_T003','ID005_T004','ID005_T005','ID005_T007','ID005_T008',...
%'ID005_T009','ID005_T010','112_07172017','118_07182017',};
%'ID006_T001','ID006_T002','ID006_T003','ID006_T004','ID006_T005','ID006_T006','ID006_T007','ID006_T008','ID006_T009','ID006_T010'}%,'I'ID005_T001','ID005_T002','ID005_T003','ID005_T004','ID005_T005','ID005_T007','ID005_T008','ID005_T012','ID005_T013','ID005_T014',...%'ID005_T011','ID005_T001','ID005_T002','ID005_T003','ID005_T004','ID005_T005','ID005_T007','ID005_T008',...
%     'ID005_T009','ID005_T010','112_07172017','118_07182017', '106_07142017','023','028','ID001_T001','ID001_T002','ID001_T003','ID001_T004','ID001_T006','ID001_T007', ...
%     'ID001_T008','ID001_T009','ID001_T010','ID001_T011','ID001_T012','ID001_T013',...
%     'ID001_T014','ID001_T017', ...
%     'ID002_T001','ID002_T002','ID002_T003','ID002_T004','ID002_T010', ...
%     'ID002_T012','ID002_T015','ID002_T016','ID002_T017','ID002_T013', ...%
%    'ID002_T019','ID002_T020', ...%'ID002_T021','ID005_T015',,'ID001_T015' 'ID002_T018',
%     'ID003_T001','ID003_T002','ID003_T003', ...
%     'ID003_T004','ID003_T010'D003_T014'
% 
data = [];
label = [];
tic;
%        'event_checklist_path',checkFile ,...
for i =1:length(videoName)
    sprintf('%s is processing!',videoName{i})
    input_path = 'D:\xishuaip\TRI\Project\Dataset';
    out_path = 'D:\xishuaip\TRI\Project\Intersection_New';
    checkFile = sprintf('%s/%s/%s_intersection.list',input_path,videoName{i}, videoName{i});
%           'event_checklist_path',checkFile ,...
    seq_y_path = sprintf('%s/%s/seq_y_10.list',out_path, videoName{i});
%     'event_checklist_path',checkFile ,...
    Mdata = Maneuverdata('data_id', videoName{i},...
        'input_path',input_path,...
        'seq_y_path',seq_y_path,...
        'log_field', {'time','speed','GPS_long','GPS_lat','GPS_heading','vector_accel','distance'},...
        'out_dir',out_path,...
         'event_field', {'Intersection','TrafficLight','TurnRight','TurnLeft'},...
         'start_padding',0.045,...
         'event_checklist_path',checkFile ,...
         'last_padding',0.025,...
         'gene_feature', {'Curvature','VGGObject','Heading','Speed'},...%,'VGGScene'
         'load_feature', {'Curvature','VGGObject','Heading','Speed'},...
         'samplingStep',0.002,...  
         'samplingProperty','distance',...    
         'seq_window',0.03,...
         'seq_step', 0.01...
        );
   
    %[seqX,seqY] = Mdata.EventSequence();
    Mdata.WriteEvent();
%     data = [data;seqX];
%     label = [label;seqY];
end
 toc;
% %%%%%%%%
% [trainD,trainL,testD, testL,testIndex] = Balancesplit(data,label.eventType,0.7);
% % fprintf('Ite:%d \n', n);
% testLabel = label(testIndex,:);
% net = ManeuversNet(trainD, trainL,400,500);
% y = classify(net,testD );
% [testAna, accuracy] = generateAccuracy(double(testL)-1,double(y)-1)
% corIndex= find( (double(testL) - double(y)) == 0 & (double(testL)-1) > 0 );
% corInf = testLabel(corIndex,:);
% 
% fprintf('sequence end to event end(time): %f \n', mean(corInf.seqEnd2eventEnd_time  ) );
% fprintf('sequence end to event end(distance): %f \n', mean(corInf.seqEnd2eventEnd_distance ) );
% fprintf('sequence end to begin end(time): %f \n', mean(corInf.seqEnd2eventBeg_time ) );
% fprintf('sequence end to begin end(distance): %f \n', mean(corInf.seqEnd2eventBeg_distance));
% % SEER_time,SEER_dis
% fprintf('sequence end to event end(time, Radio): %f \n', mean(corInf.SEER_time  ) );
% fprintf('sequence end to event end(distance, Radio): %f \n', mean(corInf.SEER_dis ) );



% corInf = ;
% fprintf('sequence end to event end(time): %f \n', mean(corInf.seqEnd2eventEnd_time) );
% fprintf('sequence end to event end(distance): %f \n', mean(corInf.seqEnd2eventEnd_distance) );
% fprintf('sequence end to begin end(time): %f \n', mean(corInf.seqEnd2eventBeg_time) );
% fprintf('sequence end to begin end(distance): %f \n', mean(corInf.seqEnd2eventBeg_distance));








