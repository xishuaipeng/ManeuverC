function mat2np(data_dir,data_id, road_path, feature_fled)
disp('Feature used');
disp(feature_fled);
mat_path = fullfile(data_dir, data_id);
mat_list = dir(mat_path);
data_file ='';
lable_file = '';
for i = 1: size(mat_list,1)
    file_name = mat_list(i).name;
    if contains(file_name, 'mat') & contains(file_name, 'seqX')
        data_file = file_name;
    end
    if contains(file_name, 'mat') & contains(file_name, 'seqY')
        lable_file = file_name;
    end       
end
save_dir = fullfile(data_dir,data_id, 'Pydata');

if (~strcmp(data_file,''))& (~strcmp(lable_file,''))
    load(fullfile(mat_path,data_file));% seqFeature
    load(fullfile(mat_path,lable_file )); %sequenceZ
    event_id = unique(sequenceZ.eventIndex);
    n_event = length(event_id);
    frame_index = [];
    wait_bar = waitbar(0, 'Writing Mat','Name','Message' );
    for i = 1: n_event
        waitbar(i/n_event, wait_bar, sprintf('Writing Mat : %d/%d',i,n_event) );
        sequence_index = find( sequenceZ.eventIndex == event_id(i));
        n_sequence = length(sequence_index);
        event_dir = fullfile(save_dir,sprintf('%d',i));
        mkdir_if_not_exist(event_dir);
        for j =1:n_sequence
            seq_struct = struct();
            data = seqFeature(sequence_index(j), feature_fled);
            data = cell2mat(table2array(data));
            seq_struct.signal = data;
            seq_struct.frame =  seqFeature.Frame{sequence_index(j)};
            frame_index = [frame_index; seq_struct.frame];
            seq_struct.path = save_dir;
            seq_struct.label = table2struct(sequenceZ(sequence_index(j),:));
            seq_struct.session = data_id;
            save(fullfile(event_dir,sprintf('%d.mat',j)), 'seq_struct','-v7.3');
        end
    end
    close (wait_bar) 
    frame_id = unique(frame_index);
    front_reader = VideoReader(road_path);
    driver_reader = VideoReader(replace(road_path,'video','driver'));
    wait_bar = waitbar(0, 'Writing image','Name','Message' );
    for i= 1:length(frame_id)
        waitbar(i/length(frame_id), wait_bar, sprintf('Writing image : %d/%d',i,length(frame_id)) );
        img_path = fullfile(save_dir,sprintf('%d_f.jpg',frame_id(i)));
        if ~exist(img_path, 'file')
        frame = read(front_reader, frame_id(i));
        imwrite(frame,img_path); 
        end
        img_path = fullfile(save_dir,sprintf('%d_d.jpg',frame_id(i)));
        if ~exist(img_path, 'file')
        frame = read(driver_reader, frame_id(i));
        imwrite(frame,img_path);  
        end
       
    end
   close (wait_bar)  
%     event_id = unique(event_id);
    
    
    
%     x = table2struct(seqFeature);
%     y = table2struct(sequenceZ);
%     save(fullfile(save_dir,'seqX'), 'x','-v7.3');
%     save(fullfile(save_dir,'seqY'), 'y','-v7.3');
end
end




% load(sample_data_path);% seqFeature
%     load(sample_label_path); %sequenceZ
%     event_id = sequenceZ.eventIndex;
%     event_id = unique(event_id);
%     n_event = length(event_id);  
%     for i = 1: n_event
%         sequence_index = find( sequenceZ.eventIndex == event_id(i));
%         n_sequence = length(sequence_index);
%         for j = 1: n_sequence
%             seq_path = fullfile(save_dir, sprintf('%d_%d_signal.mat',i,j));
%             x = seqFeature(sequence_index(j), 4:end);
%             x = table2array(x);
%             x = cell2mat(x);
%             save(seq_path,'x');
%             start_frame = seqFeature{sequence_index(j),1};
%             last_frame = seqFeature{sequence_index(j),2};
%             image = start_frame: last_frame;
% %             for k = start_frame: last_frame
% %                 
% %             end
%             seq_path = fullfile(save_dir, sprintf('%d_%d_image.mat',i,j));
%             save(seq_path,'image');
%             y = sequenceZ(sequence_index(j),:);
%             seq_path = fullfile(save_dir, sprintf('%d_%d_label.mat',i,j));
%             save(seq_path,'y');  
%         end
%         
%     end  


