clear,clc;
close all;
addpath('./DataClass', './DataClass/OBDExtraction','./Feature', './Evaluation','./Train','./Util')
videoName = {'ID001_T001','ID001_T002',...
   };
data = [];
label = [];
tic;
for i =1:length(videoName)
    sprintf('%s is processing!',videoName{i})
    input_path = '..\..\Dataset';
    out_path = '..\..\Out';
    checkFile = sprintf('%s/%s/%s.list',input_path,videoName{i}, videoName{i});
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
         'start_padding',3,...
         'last_padding',3,...
         'event_checklist_path',checkFile ,...
         'gene_feature', {'time','speed','GPS_long','GPS_lat','GPS_heading','distance','Steering_angle','vector_accel','vert_accel','Throttle_angle'},...%,'VGGScene''VGGScene',
         'load_feature', {'speed','GPS_long','GPS_lat','GPS_heading','distance','Steering_angle','vector_accel','vert_accel','Throttle_angle'},...%
         'samplingStep',0.1,...  
         'samplingProperty','time',...    
         'seq_window',3,...
         'seq_step', 1);
    %
    [seqX,seqY] = Mdata.EventSequence(i);
    mat2np(out_path, videoName{i}, Mdata.localParameters.Results.videoPath, Mdata.localParameters.Results.load_feature)
    %Mdata.WriteEvent();
    %[statis ]= Mdata.eventSta(i);
    data = [data;seqX];
    label = [label;seqY];
end
toc;
%%%%%%%%
















