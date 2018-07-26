
%%%%%%%%%%%%%%%%%%%%%Experiment 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear,clc;
close all;
addpath './OBDExtraction';
addpath './DataClass';
addpath './Feature'
addpath './Feature/VGG16'
addpath './Net'
% videoName = {'ID001_T001','ID001_T002',...
%     'ID001_T003','ID001_T004','ID001_T006','ID001_T007', ...
%     'ID001_T008','ID001_T009','ID001_T010','ID001_T011','ID001_T012','ID001_T013',...
%     'ID001_T014', 'ID001_T017', ...
%     'ID002_T001','ID002_T002','ID002_T003','ID002_T004','ID002_T010', ...
%     'ID002_T012','ID002_T015','ID002_T016','ID002_T017','ID002_T018', ...
%     'ID002_T019','ID002_T020' ...
%     'ID003_T001','ID003_T002','ID003_T003','ID003_T004','ID003_T010',...
%     'ID005_T001','ID005_T002','ID005_T003','ID005_T004','ID005_T005','ID005_T007','ID005_T008',...
%     'ID005_T009','ID005_T010','ID005_T012','ID005_T013','ID005_T014','ID005_T015'...
%     'ID006_T001','ID006_T002','ID006_T003','ID006_T004','ID006_T006','ID006_T007',...
%     'ID006_T009','ID006_T010',...
%    }; 


%'ID001_T005' No Label
%'ID001_T006' Heavy Snow
%'ID001_T007' Heavy fog and Snow
%'ID001_T008' Heavy Snow and camera occulated
input_path = 'D:\xishuaip\TRI\Project\Dataset';
out_path = 'D:\xishuaip\TRI\Project\relabel';
dir_list = dir(input_path);
videoName ={};
case_num = 1;
for i=1:size(dir_list,1)
    folder_name = dir_list(i).name;
%     if strcmp(folder_name,'ID001_T005')|strcmp(folder_name,'ID001_T006')...
%             |strcmp(folder_name,'ID001_T007')|strcmp(folder_name,'ID001_T008')...
%             |strcmp(folder_name,'ID001_T020')
%         continue;  
%     end
    if length(strfind(folder_name,'ID002'))> 0 & isdir(fullfile(input_path, folder_name))
        
        videoName{case_num} = folder_name;
        case_num = case_num+1;
    end  
end

data = [];
label = [];
tic;
for i =1:length(videoName)
    sprintf('%s is processing!',videoName{i})

    checkFile = sprintf('%s/%s/%s.list',input_path,videoName{i}, videoName{i});        %'event_checklist_path',checkFile,...
    disp( remove_mat(fullfile(out_path,videoName{i}),{'vggScene.mat';'vggObject.mat';'seqX_0.050000_0.004000_0.100000_0.020000.mat';'seqY_0.050000_0.004000_0.100000_0.020000.mat';'seqX_0.030000_0.006000_0.100000_0.020000.mat';'seqY_0.030000_0.006000_0.100000_0.020000.mat';'seqX_0.030000_0.010000_0.100000_0.020000.mat';'seqY_0.030000_0.010000_0.100000_0.020000.mat'} ));
    Mdata = Maneuverdata('data_id', videoName{i},...
        'input_path',input_path,...
        'log_field', {'time','speed','GPS_long','GPS_lat','GPS_heading','distance','Steering_angle','vector_accel','vert_accel','Throttle_angle'},...
         'out_dir',out_path,...
         'event_checklist_path',checkFile,...
         'start_padding',5,...
         'last_padding',5,...
         'gene_feature', {'Curvature','VGGObject','Heading','Speed','Steering_angle','vector_accel','vert_accel','Throttle_angle'},...%,'VGGScene''VGGScene',
         'load_feature', {'Curvature','VGGObject','Heading','Speed','Steering_angle','vector_accel','vert_accel','Throttle_angle'},...%
         'samplingStep',0.1,...  
         'samplingProperty','time',...    
         'seq_window',0.05,...
         'seq_step', 0.004...
        );
    [Mdata, sample_data] = Mdata.resampLogdata();
    temp = Mdata.resample_event_from_data(sample_data);
    %event_frame = Mdata.load_resample_event(sample_data);
    Mdata.WriteEvent();
    %[seqX,seqY] = Mdata.EventSequence(i);
    %[statis ]= Mdata.eventSta(i);
%     data = [data;seqX];
%     label = [label;seqY];
end
toc;
%%%%%%%%
[trainD,trainL,testD, testL,trainIndex, testIndex] = EventSplit(data,label,0.8);
% fprintf('Ite:%d \n', n);
net = ManeuversNet(trainD, trainL,400,500);
%
display('training');
y = predict(net, trainD );
[time_dis_ana,TP]  = eventAccuracy(y,label(trainIndex,:), 0.8);
time_dis_ana

% only for event accuracy
display('testing');
y = predict(net, testD);
[time_dis_ana,TP] = eventAccuracy(y,label(testIndex,:), 0.8);
time_dis_ana

%


















