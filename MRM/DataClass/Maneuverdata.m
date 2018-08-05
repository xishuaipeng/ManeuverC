classdef Maneuverdata < Dataset 
%--------------------------------------------------------------------------
% Maneuverdata: this is the class inherit dataset, including all the
% methods to process maneuvering detection data
%
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% @author: Xishuai Peng
% @email: xishuaip@umich.edu
% @date: May.30.2018
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%--------------------------------------------------------------------------    
    methods(Static)
        function y = LabelManeuver(eventFrame, x)
            [r,c] = size(x);
            frame = zeros(r,2);
            for i = 1:r
               frame(i,:) = [min(x{i,1}(:,1)), max(x{i,1}(:,1))]; 
            end
            y = zeros(r,1); 
            eventNum = size(eventFrame,1);
            for j = 1:eventNum
                  event_start = eventFrame(j,1);
                  event_last =  eventFrame(j,2);
                  event = find(eventFrame(j,3:end)==1);
                for i= 1:r
                    seq_start = min(x{i}(:,1));
                    seq_last = max(x{i}(:,1));     
                    start = max(event_start, seq_start);
                    last = min(event_last, seq_last);
                    duration = last - start;
                    radio_event = duration /(event_last - event_start);
                    radio_seq  =  duration /(seq_last - seq_start);
                    radio = max(radio_event,radio_seq);
                    if radio > 0.7
                        y(i) = event;
                    end
                end
            end   
        end

        function encodeevent(fid, cur_dir, score,descript)
             fwrite(fid, sprintf('%s,%f,%s \n',cur_dir,score,descript));  
        end
        
        function sample_event = event_statistic(sample_event, sample_data, event_field )
                num_event = size(sample_event,1);
                dataframe = sample_data.frame;
                maxFrame =  max(dataframe);
                for eventNum = 1: num_event  
                    startFrame = sample_event.StartFrame(eventNum);
                    endFrame = sample_event.EndFrame(eventNum);
                    eventIndex = find(sample_event{eventNum,event_field} == 1);
                    compIndex = min(1,dataframe - startFrame);
                    compIndex(compIndex== 1) = - maxFrame;
                    [~, startIdx_] = max(compIndex);
                    %end index
                    compIndex = max(-1,dataframe - endFrame);
                    compIndex(compIndex == -1) =  maxFrame;
                    [~, endIdx_] = min(compIndex);
                    sample_event.StartsampleIndex(eventNum) = startIdx_;
                    sample_event.LastsampleIndex(eventNum) = endIdx_; 
                    sample_event.StartsampleFrme(eventNum) = dataframe(startIdx_);
                    sample_event.LastsampleFrame(eventNum) = dataframe(endIdx_); 
                    sample_event.Durationtime(eventNum)  = sample_data.time(endIdx_) - sample_data.time(startIdx_);
                    sample_event.Durationdistance(eventNum)  =  sample_data.distance(endIdx_) - sample_data.distance(startIdx_);
                    sample_event.Type(eventNum) = eventIndex;
%                     sample_event.Index(eventNum) = eventNum;
                end  
            
        end
        function [cur_dir, score,descript,sindex, lindex] = decodeevent(line_ex)
            component = split(line_ex,',');
            cur_dir = component{1};
            score = str2num(component{2});
            descript = component{3};
            if length(component)==3
                sindex = -1;
                lindex = -1;
            elseif length(component)==5
                sindex = str2num(component{4});
                lindex = str2num(component{5});
            end
        end

    end
    methods
        function obj = Maneuverdata(varargin)
            %--------------------------------------------------------------
            % MANEUVERDATA: this function is a constructor, adding and
            % initializing the class local parameters.
            %
            %   Function Signature:
            %         obj = Maneuverdata(varargin)
            %--------------------------------------------------------------            
            obj@Dataset()
            obj.localParameters.addParamValue('seq_window', 0.05);
            obj.localParameters.addParamValue('seq_step', 0.01);
            
            obj.localParameters.addParamValue('gene_feature', {'Curvature','VGGObject','Heading','Speed','Steering_angle','vector_accel','vert_accel','Throttle_angle'});
            obj.localParameters.addParamValue('load_feature', {'Curvature','VGGObject','Heading','Speed'});
            
            obj.localParameters.addParamValue('start_padding', 0.1);
            obj.localParameters.addParamValue('last_padding', 0.02);
            
            obj.localParameters.addParamValue('out_dir', '');
            obj.localParameters.addParamValue('syn_file_path','');
            obj.localParameters.addParamValue('sample_data_path', '');
            obj.localParameters.addParamValue('sample_event_path','');
            obj.localParameters.addParamValue('event_checklist_path','');
            obj.localParameters.addParamValue('valid_sample_event_path','');  
            obj.localParameters.addParamValue('event_data_dir',''); 
            obj.localParameters.addParamValue('seq_y_path',''); 
            obj.localParameters.addParamValue('seq_x_path',''); 
            obj.Initialization(varargin{:});          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%update parameters
            input_path = obj.localParameters.Results.input_path;
            data_id = obj.localParameters.Results.data_id;
            if isempty(obj.localParameters.Results.out_dir);obj.localParameters = obj.update...
                    (obj.localParameters,'out_dir', sprintf('%s/processed_data',input_path));end
            mkdir_if_not_exist(fullfile(obj.localParameters.Results.out_dir,data_id));
            
            if isempty(obj.localParameters.Results.syn_file_path);obj.localParameters = obj.update...
                    (obj.localParameters,'syn_file_path', sprintf('%s/sycnFile.txt',input_path));end

            out_dir = obj.localParameters.Results.out_dir;
            samplingProperty = obj.localParameters.Results.samplingProperty;
            samplingStep = obj.localParameters.Results.samplingStep;
            if isempty(obj.localParameters.Results.sample_data_path);obj.localParameters = obj.update...
                    (obj.localParameters,'sample_data_path', sprintf('%s/%s/sampleData_%s_%.3f.mat',out_dir,data_id,samplingProperty,samplingStep));end
            if isempty(obj.localParameters.Results.sample_event_path);obj.localParameters = obj.update...
                    (obj.localParameters,'sample_event_path', sprintf('%s/%s/sampleEvent_%s_%.3f.mat',out_dir,data_id,samplingProperty,samplingStep));end
            if isempty(obj.localParameters.Results.valid_sample_event_path);obj.localParameters = obj.update...
                    (obj.localParameters,'valid_sample_event_path', sprintf('%s/%s/ValidsampleEvent_%s_%.3f.mat',out_dir,data_id,samplingProperty,samplingStep));end
            
            if isempty(obj.localParameters.Results.event_checklist_path);obj.localParameters = obj.update...
                    (obj.localParameters,'event_checklist_path', sprintf('%s/%s/%s.list',out_dir,data_id,data_id));end
            
            if isempty(obj.localParameters.Results.event_data_dir);obj.localParameters = obj.update...
                (obj.localParameters,'event_data_dir', sprintf('%s/%s/Event',out_dir,data_id));end
            seq_window = obj.localParameters.Results.seq_window;
            seq_step = obj.localParameters.Results.seq_step;
            
           start_padding = obj.localParameters.Results.start_padding;
            last_padding = obj.localParameters.Results.last_padding;
            
            if isempty(obj.localParameters.Results.seq_y_path);obj.localParameters = obj.update...
                (obj.localParameters,'seq_y_path', sprintf('%s/%s/seqY_%f_%f_%f_%f.mat',out_dir,data_id,seq_window,seq_step,start_padding, last_padding ));end     
            if isempty(obj.localParameters.Results.seq_x_path);obj.localParameters = obj.update...
                (obj.localParameters,'seq_x_path', sprintf('%s/%s/seqX_%f_%f_%f_%f.mat',out_dir,data_id,seq_window,seq_step,start_padding, last_padding));end     
        end
        
        function [obj, sample_data] = resampLogdata(obj)
            %--------------------------------------------------------------
            % RESAMPLOGDATA: this function is to resample log data inherit
            % from class dataset resampLogdata function
            %
            %   Function Signature:
            %         [obj, sample_data] = resampLogdata(obj)
            %--------------------------------------------------------------
            sample_data_path = obj.localParameters.Results.sample_data_path;
            videoPath = obj.localParameters.Results.videoPath;
            vidObj = VideoReader(videoPath);
            obj.localParameters = obj.update(obj.localParameters,'frameRate',vidObj.FrameRate); 
            maxFrame = vidObj.NumberOfFrames;
            [obj,logData] = obj.readLogdata();
            syn_file_path = obj.localParameters.Results.syn_file_path;
            data_id = obj.localParameters.Results.data_id;
            timeDelayforVideo = checkIfSync(logData, data_id, syn_file_path);
            obj.localParameters = obj.update(obj.localParameters,'timeDelayforVideo',timeDelayforVideo);     
            if exist(sample_data_path,'file')
                clear sample_data
                load(sample_data_path,'sample_data');
            else
                [obj,sample_data] = resampLogdata@Dataset(obj, logData);
                sample_data(sample_data.frame>maxFrame,:)=[];
                sample_data(sample_data.frame<1,:)=[];
                save(sample_data_path,'sample_data' );
            end
        end
        
        
        function sample_event = resample_event_from_data(obj, data)
            %--------------------------------------------------------------
            % RESAMPLE_EVENT_FROM_DATA: change the event time to sample
            % index and sample frame
            %
            %   Function Signature:
            %         sample_event = resample_event_from_data(obj, data)
            %--------------------------------------------------------------            
            sample_event_path = obj.localParameters.Results.sample_event_path;
            event_field = obj.localParameters.Results.event_field;
            if exist(sample_event_path,'file')
                clear sample_event
                load(sample_event_path,'sample_event');
            else
                %function part 
                sample_event= obj.readEvent();
                index = (1:size(sample_event,1));
                sample_event = obj.event_statistic(sample_event, data, event_field );
                sample_event.Index = index';
                save(sample_event_path,'sample_event');
            end
            event_checklist_path = obj.localParameters.Results.event_checklist_path;
            if ~exist(event_checklist_path,'file')
                list_writer = fopen(event_checklist_path,'w+');
                try
                    num_event = size(sample_event,1);
                    for eventNum = 1: num_event 
                        eventIndex = sample_event.Type(eventNum);
                        cur_dir = sprintf('%d_%s',eventNum, event_field{eventIndex});
                        obj.encodeevent(list_writer, cur_dir, 1,'NEEDCHECK');  
                    end
                catch
                    fclose(list_writer);
                    display('write list file wrong!')
                    return; 
                end
                fclose(list_writer);
            end
        end  
        
        function sample_event = load_resample_event(obj, data)
            %--------------------------------------------------------------
            % LOAD_RESAMPLE_EVENT: this function is to save
            % valid_sample_event data according to the checklist.list file
            % to load the sample event.
            %
            %   Function Signature:
            %         sample_event = load_resample_event(obj,sample_data)
            %--------------------------------------------------------------                
            valid_sample_event_path = obj.localParameters.Results.valid_sample_event_path;
            event_field = obj.localParameters.Results.event_field;
            if exist(valid_sample_event_path,'file')
                clear sample_event;
                load(valid_sample_event_path,'sample_event');
            else
                event_checklist_path = obj.localParameters.Results.event_checklist_path;
                sample_event_path = obj.localParameters.Results.sample_event_path;
                if ~exist(event_checklist_path,'file') | ~exist(sample_event_path,'file')
                    sample_event = obj.resample_event_from_data(data);
                else
                    clear sample_event
                    load(sample_event_path,'sample_event'); 
                end
                valid_reader = fopen(event_checklist_path,'r');
                index = 0;
                deleteRow = [];
                while ~feof(valid_reader)
                    tline = fgetl(valid_reader);
                    if isempty(tline)
                        continue;
                    end
                    index = index +1;
                    [dirStr,score,description,sindex, lindex] = obj.decodeevent(tline);
                    if score < 0.5
                        deleteRow = [deleteRow,index];
                        continue;
                    else
                        dirStr = replace(dirStr,'Event\','');
                        description = strip(description);
                        if strcmp(description,'NEEDCHECK')
                            display([dirStr,' is not chenked']);
                        end  
                        if lindex~=-1 & sindex~=-1
                           sample_event.StartFrame(index) = sindex;
                           sample_event.EndFrame(index) = lindex;
                        end
                    end
                end
                 sample_event(deleteRow,:) = [];
                 sample_event = obj.event_statistic(sample_event, data, event_field );
                 save(valid_sample_event_path,'sample_event');   
                 fclose(valid_reader);
            end
        end
        
          
       function sequenceZ = sequence_from_event(obj, sample_data, index)
            %--------------------------------------------------------------
            % SEQUENCE_FROM_EVENT: this function is to save sequenceZ,
            % which stores the sequence information eventType, start index
            % and end index
            %
            %   Function Signature:
            %         sequenceZ = sequence_from_event(obj, sample_data)
            %--------------------------------------------------------------   
            seq_y_path =  obj.localParameters.Results.seq_y_path;
            if exist(seq_y_path,'file')
                clear sequenceZ
                load(seq_y_path,'sequenceZ');
             else
                event_frame = obj.load_resample_event(sample_data);
                num_event = size(event_frame,1);
                start_padding = obj.localParameters.Results.start_padding;
                last_padding = obj.localParameters.Results.last_padding;
                samplingStep = obj.localParameters.Results.samplingStep;
                start_padding = start_padding/samplingStep;
                last_padding = last_padding/samplingStep;
                
                seq_window = obj.localParameters.Results.seq_window;
                seq_step = obj.localParameters.Results.seq_step;
                samplingStep = obj.localParameters.Results.samplingStep;
                
                seq_window = floor(seq_window/samplingStep);
                seq_step = floor(seq_step/samplingStep);
                
                maxIndex = size(sample_data,1);
                sequence_index = [];
                
%                 sequence_id = {}
                sample_data.label = zeros(length(sample_data.frame),1);
                for i =1:num_event
                    startIndex = event_frame.StartsampleIndex(i);
                    lastIndex =  event_frame.LastsampleIndex(i);
                    eventLabel = event_frame.Type(i);
                    sample_data.label(startIndex: lastIndex) = eventLabel; 
                end
               
                for i = 1 : num_event
                    startIndex = event_frame.StartsampleIndex(i);
                    data_id =    obj.localParameters.Results.data_id;
                    lastIndex =  event_frame.LastsampleIndex(i);
                    if lastIndex - startIndex > 300
                       display(sprintf('Event %d (%d)is too long,please recheck!',i,(last_start_index - startPadding))); 
                    end
                    eventIndex = event_frame.Index(i);
                    eventLabel = event_frame.Type(i);
                    startPadding = max(1,round(startIndex - start_padding));
                    lastPadding = min(maxIndex,round(lastIndex + last_padding)); 
                    last_start_index = (lastPadding - seq_window);
                    sequence_start = [startPadding:seq_step:last_start_index];
                    if isempty(sequence_start)
                        display(sprintf('Event %d is too short,please recheck!',eventIndex));
                        continue;
                    end

                    if ~(sequence_start(end)==last_start_index)
                        sequence_start = [sequence_start,last_start_index];
                    end

                    for j = 1: length(sequence_start)
                        seq_start = sequence_start(j);
                        seq_end = sequence_start(j) + seq_window ;
                        sequence_id    = index * 1000 +  eventIndex;    
                        sequenceLabel =  find( sample_data.label(seq_start:seq_end)~=0 & sample_data.label(seq_start:seq_end)~=eventLabel);
                        if length(sequenceLabel) >0
                            sequenceLabel = -1;
                            continue;
                        else
                            sequenceLabel = eventLabel;
                        end
                        sequence_index = [sequence_index ; sequence_id, eventIndex, eventLabel,sequenceLabel,...
                        sample_data.time(seq_start) ,sample_data.time(seq_end),...
                        sample_data.time(startIndex) ,sample_data.time(lastIndex),...
                        sample_data.distance(seq_start) ,sample_data.distance(seq_end),...
                        sample_data.distance(startIndex) ,sample_data.distance(lastIndex), seq_start, seq_end];
                    end
                end
                sequenceZ = array2table(sequence_index);
                sequenceZ.Properties.VariableNames = {'sequenceIndex','eventIndex','eventLabel','sequenceLabel','seq_start_time','seq_end_time',...
                    'event_start_time','event_end_time',...
                    'seq_start_distance','seq_end_distance',...
                    'event_start_distance','event_end_distance','seq_start','seq_end'};  
                save(seq_y_path, 'sequenceZ');
            end
            sequenceZ = obj.label_process(sequenceZ, 'prediction_distance') ;
        end
         
        function WriteEvent(obj)
            %--------------------------------------------------------------
            % WRITEEVENT: this function is to write event video and
            % sample data and save them.
            %
            %   Function Signature:
            %         WriteEvent(obj)
            %--------------------------------------------------------------      
            [obj, sample_data] = obj.resampLogdata();
            data_id = obj.localParameters.Results.data_id;
            event_field = obj.localParameters.Results.event_field;
            session_duation = sample_data.time(end) - sample_data.time(1) ;
            session_distance = 1000*(sample_data.distance(end) - sample_data.distance(1)) ;
            start_padding = obj.localParameters.Results.start_padding;
            last_padding = obj.localParameters.Results.last_padding;
            samplingStep = obj.localParameters.Results.samplingStep;
            start_padding = start_padding/samplingStep;
            last_padding = last_padding/samplingStep;
            event_checklist_path = obj.localParameters.Results.event_checklist_path;
            videoPath = obj.localParameters.Results.videoPath;
            event_data_dir = obj.localParameters.Results.event_data_dir;
            vidObj = VideoReader(videoPath);
            frameRate = obj.localParameters.Results.frameRate;
%             sample_event = obj.resample_event_from_data();
            sample_event = obj.load_resample_event(sample_data); 
            valid_reader = fopen(event_checklist_path,'r');
            index = 0;
            valid_event_num = size(sample_event,1) ;
            for  i = 1: valid_event_num
                    event_index = sample_event.Index(i);
                    event_type = sample_event.Type(i);
                    savePath = fullfile( event_data_dir, sprintf('%d_%d', event_index, event_type));
                    mkdir_if_not_exist(savePath) 
                    matPath = fullfile(savePath, sprintf('%d_%d_%.1f.mat',start_padding,last_padding, 1000*samplingStep));
                    if exist(matPath,'file')
                        continue;     
                    end
                    startFrame = sample_event.StartsampleIndex(i);
                    endFrame = sample_event.LastsampleIndex(i);
                    startIdx = max(1,round(startFrame - start_padding));
                    endIdx = min(length(sample_data.frame), round(endFrame + last_padding));
                    if startIdx >= endIdx
                        continue;
                    end
                    segData = sample_data(startIdx : endIdx, :);
                    save(matPath, 'segData');  
                    %write
                    segVidObj = VideoWriter(fullfile(savePath , sprintf('%d_%d_%.1f.avi',start_padding,last_padding, 1000*samplingStep)));
                    segVidObj.FrameRate = frameRate;
                    open(segVidObj);
                    segFrame= segData.frame;
                    % for experiment
                    mkdir_if_not_exist(fullfile(savePath,'frame'));
                    for frameIndex = 1 : length(segFrame) 
                        t_frame = read(vidObj,segFrame(frameIndex));
                        writeVideo(segVidObj,t_frame);
                        t_frame(:, floor(size(t_frame,2)/4),1)=255;
                        t_frame(:, floor(3*size(t_frame,2)/4),1)=255;
                        t_frame(:, floor(size(t_frame,2)/6),2)=255;
                        t_frame(:, floor(5*size(t_frame,2)/6),2)=255;
                        if (frameIndex > start_padding) & ((length(segFrame) - frameIndex) > last_padding)
                            imwrite(t_frame, fullfile(savePath,sprintf('%s/%d_l.jpg', 'frame',segFrame(frameIndex))));
                        else
                            imwrite(t_frame, fullfile(savePath,sprintf('%s/%d.jpg', 'frame',segFrame(frameIndex))));
                        end
                    end    
                     close(segVidObj);  
            end
             fclose(valid_reader);
             disp(sprintf('%s, duration:%f(s), distance:%f(m),#event:%d',data_id,session_duation,session_distance,sum(valid_event_num)));
%              str = '';
%              for i = 1:size(event_field,2)
%                 str = sprintf(' %s  %s,%d',str, event_field{i},valid_event_num(i));
%              end
%              disp(str);
        end
        
         function data = label_process(obj, data, task_type) 
            if strcmp(task_type,'prediction_distance')
                early_threshold = 0.5*(data.event_end_distance + data.event_start_distance) - data.seq_end_distance;
                last_threshold = data.event_end_distance -  data.seq_start_distance;
                data.distance_to_maneuver = early_threshold;
                data.distance_to_end = last_threshold;
                data.sequenceLabel(early_threshold > 50/1000|last_threshold < 0) = 0;
            elseif strcmp(task_type,'recognition')
                early_threshold = 0.5*(data.event_end_time + data.event_start_time) - data.seq_end_time;
                last_threshold = data.event_end_time -  data.seq_start_time;
                data.sequenceLabel(early_threshold>0|last_threshold<0) = 0;   
            else
                disp('Can not find label criterion! ')
            end  
         end
        
        function [seqX,sequenceZ] = EventSequence(obj, index)
            %--------------------------------------------------------------
            % EVENTSEQUENCE: this function is the main function to process
            % the data, storing the resample data and extract features.
            %
            %   Function Signature:
            %         [seqX,sequenceZ] = EventSequence(obj)
            %--------------------------------------------------------------             
            seq_x_path = obj.localParameters.Results.seq_x_path;
            [obj, sample_data] = obj.resampLogdata();
            data_id =  obj.localParameters.Results.data_id;
            out_dir = obj.localParameters.Results.out_dir;
            gene_feature = obj.localParameters.Results.gene_feature;
            load_feature = obj.localParameters.Results.load_feature;
            videoPath = obj.localParameters.Results.videoPath;
            sequenceZ = obj.sequence_from_event(sample_data, index);
%             if sequence       
            if exist(seq_x_path)
                clear seqFeature;
                load(seq_x_path,'seqFeature');
            else
                seqFeature = ExtraSequenceFeature(sample_data, gene_feature, videoPath,sequenceZ.seq_start, sequenceZ.seq_end,fullfile(out_dir,data_id));
                save(seq_x_path,'seqFeature','-v7.3');
            end
            seqFeature = table2array( seqFeature(:,load_feature));
            nCase = size(seqFeature,1);
            seqX = cell(nCase,1);
            for i = 1:nCase; seqX(i,1)={cat(2,seqFeature{i,:})}; end
            seqX = cellfun(@(x) x', seqX,'UniformOutput',false);
        end
        
        
        
       
        
    end
    
end