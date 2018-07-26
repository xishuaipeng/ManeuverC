function OBD = extractOBDinfo(OBD_filepath,fid_log, targetParams)
    fid = fopen(OBD_filepath);
    OBD.dataRate = 100;
    [OBD.startDate, OBD.startTime] = getOBDstartDateTime(fid);
    OBD.initLocation = getOBDinitLocation(fid);
    headerString = fgetl(fid);
    OBD.params = getSigParamsHeader(headerString);
    OBD.targetParams = targetParams;
%     OBD.targetParams = {'time [s]', 'speed [mph]', 'GPS long [degs]', ...
%         'GPS lat [degs]', 'GPS heading [degs]', ...
%         'long accel [g]', 'lat accel [g]','vector accel [g]', ...
%         'vert accel [g]', 'distance [km]', 'position X [m]', ...
%         'position Y [m]','GPS altitude [m]'};
    [OBD.data,OBD.nonzero_ind] = getOBDdata(OBD.targetParams, OBD.params, fid);
    fclose(fid);
   
%   numSeconds = floor(str2num(OBD.data{end, 1}));
    numSeconds = floor(size(OBD.data, 1) / 100);
    numMilliseconds = round(mod(str2num(OBD.data{end, 1}) * 100, 100));
    
    
    OBD.endTime = addtodate(datenum([OBD.startDate, ' ', OBD.startTime], ...
        'mm/dd/yyyy HH:MM:SS.FFF'), numSeconds, 'second');
    OBD.endTime = addtodate(OBD.endTime, numMilliseconds, 'millisecond');
    OBD.endTime = datestr(OBD.endTime, 'HH:MM:SS.FFF');
    fprintf(fid_log,'successfully read data collection from OBD port!\n');
    disp('successfully read data collected by bioharness!');
end