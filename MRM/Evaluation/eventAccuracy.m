function [time_dis_type, correct_index] = eventAccuracy(pre, trues, threshold)
id = trues.sequenceIndex;

[unique_id, ia, ic] = unique(id,'stable');
% id_type = true.
event_sequence = [ia,[ia(2:end)-1;length(id)]];
min_type = min( trues.sequenceLabel(:));
max_type = max( trues.sequenceLabel(:));
if min_type == 0 
    min_type=1;
end
num_type = max_type - min_type + 1;
id_num = length(ia);
TP = zeros(id_num, num_type);
NP = zeros(id_num, num_type );
time_to_maneuver = zeros(1, id_num);
dis_to_maneuver =  zeros(1, id_num);
maneuver_type = zeros(1, id_num);
correct_index = [];
correct_id = [];

for i = 1: id_num
    start = event_sequence(i,1);
    last = event_sequence(i,2);
    maneuver_type(i) =  max(trues.sequenceLabel(start:last)) ;  %major(trues.eventType(start:last), num_type);% trues.eventType(last);
    for j = start: last
        [maxV, maxI] = max(pre(j,:));
        if maxV > threshold & maxI >1
            pre_type = maxI-1 ;
        else
            continue;
        end
        if pre_type == maneuver_type(i)
            TP(i, pre_type) = 1;
            correct_index = [correct_index; (start:last)'];
            correct_id = [correct_id; i];
            time_to_maneuver(i) = 0.5*(trues.event_start_time(j) +  trues.event_end_time(j)) - trues.seq_end_time(j);%trues.seqEnd2eventBeg_time(j);
            dis_to_maneuver(i) =   0.5*(trues.event_start_distance(j) +  trues.event_end_distance(j)) - trues.seq_end_distance(j);
            break;   
        else
            NP(i,pre_type) = 1;
            break; 
        end  
    end
end
time_dis_recal_type = zeros(4, max_type - min_type +1);
% dis_type  = zros(1, max_type - min_type +1);
for i = min_type : max_type
   index = find(maneuver_type == i);
   if length(index)==0
       continue;
   end
   time_dis_recal_type(1, i - min_type +1) = mean(time_to_maneuver(index));
   time_dis_recal_type(2, i - min_type +1) = mean(dis_to_maneuver(index));
   time_dis_recal_type(3, i - min_type +1) = sum(TP(:,i))/ length(index);
   time_dis_recal_type(4, i - min_type +1) = sum(TP(:,i))/( sum(NP(:,i))+  sum(TP(:,i)));
end

time_dis_type = array2table( time_dis_recal_type);
time_dis_type.Properties.RowNames = {'Time(s)', 'Distance(km)','Recall','Precision'};
time_dis_type.Properties.VariableNames = {'RT','LT','RLC','LLC'};
all_mean  = [ mean(time_to_maneuver); mean(dis_to_maneuver); sum(TP(:))/ id_num ; sum(TP(:))/ ( sum(NP(:,i))+  sum(TP(:,i)) ) ];
event_mean  = mean(time_dis_recal_type,2);

time_dis_type.AllMean = all_mean;
time_dis_type.EventMean = event_mean;
end


function value = major(data,bin_num)
bins = zeros(1, bin_num);
num = length(data);
for i = 1 : num
   bins(data(i)) = bins(data(i)) + 1; 
end
max_num = 0;
for i = 1:bin_num
    cur = bins(i);
    if max_num < cur
        max_num = cur;
        value = i;
    end  
end
end


