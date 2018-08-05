clear,clc;
close all;
addpath('./DataClass', './DataClass/OBDExtraction','./Feature', './Evaluation','./Train','./Util')
videoName = {'ID001_T001','ID001_T002','ID001_T003','ID001_T004',...
            'ID001_T009','ID001_T010'...
};
data = [];
label = [];
tic;
gene_feature= {'Curvature','Speed','VGGObject', 'Heading'};
load_feature =  {'VGGObject'};
start_padding = 10;
out_path = '..\..\Begin_10';
input_path = '..\..\Dataset';

for i =1:length(videoName)
    sprintf('%s is processing!',videoName{i})
    checkFile = sprintf('%s/%s/%s.list',input_path,videoName{i}, videoName{i});
   % remove_mat(fullfile(out_path,videoName{i}), {'seqX_3.000000_1.000000_3.000000_3.000000.mat'; 'seqY_3.000000_1.000000_3.000000_3.000000.mat'});
%     disp( remove_mat(fullfile(out_path,videoName{i}),...
%     {'vggScene.mat';'vggObject.mat';'seqX_0.050000_0.004000_0.100000_0.020000.mat';...
%         'seqY_3.000000_1.000000_3.000000_3.000000.mat';...
%         'seqX_3.000000_1.000000_3.000000_3.000000.mat';...
%         'seqY_0.030000_0.006000_0.100000_0.020000.mat';...
%         'seqX_0.030000_0.010000_0.100000_0.020000.mat';...
%         'seqY_0.030000_0.010000_0.100000_0.020000.mat'} ));
    Mdata = Maneuverdata('data_id', videoName{i},...
        'input_path',input_path,...
        'log_field', {'time','speed','GPS_long','GPS_lat','GPS_heading','distance','Steering_angle','vector_accel','vert_accel','Throttle_angle'},...
        'out_dir',out_path,...
         'start_padding',start_padding,...
         'last_padding',3,...
         'event_checklist_path',checkFile ,...
         'gene_feature', gene_feature,...%,'VGGScene''VGGScene',
         'load_feature',load_feature,...%
         'samplingStep',0.1,...  
         'samplingProperty','time',...    
         'seq_window',3,...
         'seq_step', 1);
    %
    [seqX,seqY] = Mdata.EventSequence(i);
%     mat2np(out_path, videoName{i}, Mdata.localParameters.Results.videoPath, Mdata.localParameters.Results.load_feature)
%     Mdata.WriteEvent();
    %[statis ]= Mdata.eventSta(i);
    data = [data;seqX];
    label = [label;seqY];
end
toc;
 net = ManeuversNet(data, categorical( label.sequenceLabel),100,1000);
%
display('training');
y = predict(net, data );
for threshold = 0.6:0.1:0.8
    display(sprintf('threshold:%f',threshold));
    [time_dis_ana,TP] = eventAccuracy(y,label(:,:), threshold);
    time_dis_ana
end
   

% only for event accuracy
%%%%%%%%

videoName = {'ID001_T011','ID001_T012','ID001_T013','ID001_T014',...
            'ID001_T015','ID001_T016','ID001_T017','ID001_T018','ID001_T019'...
};
data = [];
label = [];
tic;
for i =1:length(videoName)
    sprintf('%s is processing!',videoName{i})

    checkFile = sprintf('%s/%s/%s.list',input_path,videoName{i}, videoName{i});
   % remove_mat(fullfile(out_path,videoName{i}), {'seqX_3.000000_1.000000_3.000000_3.000000.mat'; 'seqY_3.000000_1.000000_3.000000_3.000000.mat'});
%     disp( remove_mat(fullfile(out_path,videoName{i}),...
%     {'vggScene.mat';'vggObject.mat';'seqX_0.050000_0.004000_0.100000_0.020000.mat';...
%         'seqY_3.000000_1.000000_3.000000_3.000000.mat';...
%         'seqX_3.000000_1.000000_3.000000_3.000000.mat';...
%         'seqY_0.030000_0.006000_0.100000_0.020000.mat';...
%         'seqX_0.030000_0.010000_0.100000_0.020000.mat';...
%         'seqY_0.030000_0.010000_0.100000_0.020000.mat'} ));
    Mdata = Maneuverdata('data_id', videoName{i},...
        'input_path',input_path,...
        'log_field', {'time','speed','GPS_long','GPS_lat','GPS_heading','distance','Steering_angle','vector_accel','vert_accel','Throttle_angle'},...
        'out_dir',out_path,...
         'start_padding',start_padding,...
         'last_padding',3,...
         'event_checklist_path',checkFile ,...
         'gene_feature',gene_feature,...
         'load_feature',load_feature,...%
         'samplingStep',0.1,...  
         'samplingProperty','time',...    
         'seq_window',3,...
         'seq_step', 1);
    %
    [seqX,seqY] = Mdata.EventSequence(i);
%     mat2np(out_path, videoName{i}, Mdata.localParameters.Results.videoPath, Mdata.localParameters.Results.load_feature)
%     Mdata.WriteEvent();
    %[statis ]= Mdata.eventSta(i);
    data = [data;seqX];
    label = [label;seqY];
end
toc;
%
display('training');
y = predict(net, data );
for threshold = 0.6:0.1:0.8
    display(sprintf('threshold:%f',threshold));
    [time_dis_ana,TP] = eventAccuracy(y,label(:,:), threshold);
    time_dis_ana
end














