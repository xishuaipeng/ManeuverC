function [OBD_table]= TRI_GPS_extract_oneTrip(pathin, pathout,Params)
%---------------------------------------------------------------------------
%TRI_GPS_extract_oneTrip   Main program for extracting GPS data from raw 
% data
%
%   Usage:
%       [OBD_table, tripID,Path] = TRI_GPS_extract_oneTrip() 
%       [OBD_table, tripID,Path] = TRI_GPS_extract_oneTrip(filePath)
%       [OBD_table]= TRI_GPS_extract_oneTrip(pathin, pathout)
%       select the raw data(datalog.csv file) and select path to save OBD
%       data
%   @input:
%       None
%   @output:
%       None
%---------------------------------------------------------------------------
%
%---------------------------------------------------------------------------
% @author: Ruirui Liu
% @date:   Jul.20.2017
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%---------------------------------------------------------------------------

[filePath,fileName,ext] = fileparts(pathin);
filePath = [filePath '\'];
fileName = [fileName ext];
[outPath,outName,outext] = fileparts(pathout);
outPath = [outPath '\'];
outName = [outName outext];
if exist(pathout)
    clear OBD_table;
    load(pathout,'OBD_table');
    return ;
end
fid=fopen(fullfile(outPath,'data_OBD.log'),'w+');
if fid==-1
    error('cannot log data synchronization process');
end
fprintf(fid,'beginning time: %s\n',datestr(datetime));
obd_file = getOBDfile(filePath,fileName);

OBD = extractOBDinfo(obd_file,fid, Params);
% Params = {'time', 'speed', 'GPS_long', ...
%     'GPS_lat', 'GPS_heading', ...
%     'long_accel', 'lat_accel','vector_accel',...
%     'vert_accel','distance', 'position_X', ...
%     'position_Y','GPS_altitude'};
OBD_table = array2table(OBD.data);
OBD_table.Properties.VariableNames = Params(OBD.nonzero_ind);

% table(OBD.data(:,1), OBD.data(:,2), OBD.data(:,3), ...
%     OBD.data(:,4), OBD.data(:,5), OBD.data(:,6), OBD.data(:,7), ...
%     OBD.data(:,8), OBD.data(:,9), OBD.data(:,10), 'VariableNames',Params);
save(pathout,'OBD_table');
fclose(fid);
end % EOF
%