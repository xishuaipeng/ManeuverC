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
event=[];
event_id = [];
for i = 1:length(videoName)
   path =  sprintf('../process_time/%s/ValidsampleEvent_time_0.100.mat',videoName{i});
   val_event = load(path); 
   [num_e, ~] = size(val_event.sample_event);
   for j = 1:num_e
    event_id = [event_id; videoName(i)];
   end
   event = [event;table2array(val_event.sample_event)];
end
distance = event(:,12);
time = event(:,11);
dis_nor = (distance - min(distance)) /(max(distance) - min(distance));
time_nor = (time - min(time)) /(max(time) - min(time));
event_type = event(:,13);

min_lab = min(event_type);
msx_lab = max(event_type);
lrt_index = find(event_type == 1 |event_type == 2  );
lrlc_index = find(event_type == 3 |event_type == 4  );
%distance
value =     distance(lrt_index);
value_nor =  dis_nor(lrt_index);
disp (sprintf('Turn distance: mean: %f, var:%f, mean(nor):%f, ,var(nor):%f',mean(value) ,var(value),mean(value_nor) ,var(value_nor)));
value =     time(lrt_index);
value_nor =  time_nor(lrt_index);
disp(sprintf('Turn time: mean: %f, var:%f, mean(nor):%f, ,var(nor):%f',mean(value) ,var(value),mean(value_nor) ,var(value_nor)));
value =     distance(lrlc_index);
value_nor =  dis_nor(lrlc_index);
disp(sprintf('LC distance: mean: %f, var:%f, mean(nor):%f, ,var(nor):%f',mean(value) ,var(value),mean(value_nor) ,var(value_nor)));
value =     time(lrlc_index);
value_nor =  time_nor(lrlc_index);
disp(sprintf('LC time: mean: %f, var:%f, mean(nor):%f, ,var(nor):%f',mean(value) ,var(value),mean(value_nor) ,var(value_nor)));



disp(sprintf('Total diatance, mean:%f, var:%f, mean(nor):%f, var(nor):%f',mean(distance) ,var(distance),mean(dis_nor) ,var(dis_nor)));

disp(sprintf('Total time, mean:%f, var:%f, mean(nor):%f, var(nor):%f',mean(time) ,var(time),mean(time_nor) ,var(time_nor)));

