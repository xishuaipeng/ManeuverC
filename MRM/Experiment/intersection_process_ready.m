clc;
clear;
close;
addpath(genpath('../Code_3_2'))
addpath(genpath('../Dataset'))
addpath(genpath('../Intersection_New'))
addpath(genpath('../Intersection'))
videoName = {'ID005_T001','ID005_T002','ID005_T003','ID006_T001','ID006_T002','ID006_T003','ID008_T001','ID008_T002','ID008_T003'};
data = [];
label = [];

for i = 1:length(videoName)
    driver_id = split(videoName{i},'_')
    driver_id = driver_id{1,1}
    driver_id = [driver_id]
    vid = [videoName{i}]
    sprintf('%s is processing',videoName{i})
    input_path = 'D:\xishuaip\TRI\Project\Intersection_New\';
    file_path = 'D:\xishuaip\TRI\Project\Dataset\';
    output_path = 'D:\xishuaip\TRI\Project\Intersection\Intersection_Process_Vis\';
    checkfile = sprintf('%s/%s/%s_intersection.list',file_path,videoName{i}, videoName{i});
    fileID = fopen(checkfile)
    C = textscan(fileID,'%s %n %s','Delimiter',',')
    fclose(fileID)
    [new] = listprocess(C,output_path,driver_id,input_path,vid)
    
end

function [new]=listprocess(C,output_path,driver_id,input_path,vid)
[row,c] = size(C{1,1});
index = 1;
new = {};
for i = 1:row
    if isequal(C{1,2}(i,1),0)
        continue;
    end
    new{index,1} = C{1,1}(i,1);
    new{index,3} = C{1,2}(i,1);
    new{index,2} = C{1,3}(i,1);
    index = index +1;
end
new = cell2table(new);
new = new(:,1:2)
[r,c] = size(new)
for i = 1:r
    con = new(i,2);
    con = con.new2;
    con = [con];
    if strcmp(con{1},'Straight_Green ')
        file_name = new(i,1);
        source = [input_path,vid,'\Event\',file_name.new1{:},'\2.250000e+01_1.250000e+01_2.0.mat'];%
        destination = [output_path,'Straight_Green\',driver_id];
        [status]= copyfile(source,destination);
        d =  length(dir(destination)) - 2;
        d = int2str(d);
%        re = [vid,'_',d,'.mat'];
        re = [driver_id,'_',d,'.mat'];        
        ren = ['ren ',destination,'\2.250000e+01_1.250000e+01_2.0.mat ',re];
        system(ren);
    elseif isequal(con{1},'Straight_Red ')
        file_name = new(i,1);
        source = [input_path,vid,'\Event\',file_name.new1{:},'\2.250000e+01_1.250000e+01_2.0.mat'];%
        destination = [output_path,'Straight_Red\',driver_id];
        d =  length(dir(destination)) - 2+1;
        d = int2str(d);
        [status]= copyfile(source,destination);
%        re = [vid,'_',d,'.mat'];
        re = [driver_id,'_',d,'.mat'];
        ren = ['ren ',destination,'\2.250000e+01_1.250000e+01_2.0.mat ',re];
        system(ren);
    elseif isequal(con{1},'Right_Green ')
        file_name = new(i,1);
        source = [input_path,vid,'\Event\',file_name.new1{:},'\2.250000e+01_1.250000e+01_2.0.mat'];%
        destination = [output_path,'Right_Green\',driver_id];
        d =  length(dir(destination)) - 2+1;
        d = int2str(d);
        [status]= copyfile(source,destination)
%        re = [vid,'_',d,'.mat'];
        re = [driver_id,'_',d,'.mat'];
        ren = ['ren ',destination,'\2.250000e+01_1.250000e+01_2.0.mat ',re];
        system(ren);
    elseif isequal(con{1},'Right_Red ')
        file_name = new(i,1)
        source = [input_path,vid,'\Event\',file_name.new1{:},'\2.250000e+01_1.250000e+01_2.0.mat'];%
        destination = [output_path,'Right_Red\',driver_id];
        d =  length(dir(destination)) - 2+1;
        d = int2str(d);
        [status]= copyfile(source,destination)
%        re = [vid,'_',d,'.mat'];
        re = [driver_id,'_',d,'.mat'];
        ren = ['ren ',destination,'\2.250000e+01_1.250000e+01_2.0.mat ',re];
        system(ren);
    elseif isequal(con{1},'Left_Green ')
        file_name = new(i,1);
        source = [input_path,vid,'\Event\',file_name.new1{:},'\2.250000e+01_1.250000e+01_2.0.mat'];%
        destination = [output_path,'Left_Green\',driver_id];
        [status]= copyfile(source,destination);
        d =  length(dir(destination)) - 2;
        d = int2str(d);
%        re = [vid,'_',d,'.mat'];
        re = [driver_id,'_',d,'.mat'];
        ren = ['ren ',destination,'\2.250000e+01_1.250000e+01_2.0.mat ',re];
        system(ren);
    elseif isequal(con{1},'Left_Red ')
        file_name = new(i,1);
        source = [input_path,vid,'\Event\',file_name.new1{:},'\2.250000e+01_1.250000e+01_2.0.mat'];%
        destination = [output_path,'Left_Red\',driver_id];
        d =  length(dir(destination)) - 2+1;
        d = int2str(d);
        [status]= copyfile(source,destination);
%        re = [vid,'_',d,'.mat'];
        re = [driver_id,'_',d,'.mat'];
        ren = ['ren ',destination,'\2.250000e+01_1.250000e+01_2.0.mat ',re];
        system(ren);
    else 
        continue;
    end
    
end
end