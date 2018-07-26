function obd_file = getOBDfile( dataRootPath,obd_unique_str )
%recognize obd file in the data root folder
folder = dataRootPath;
dir_info = dir([folder, obd_unique_str]);
if isempty(dir_info)
    error(['cannot find OBD file with specified obd_unique_str ',obd_unique_str]);
end
if size(dir_info,1)>1
    error('more than obd file exit, check it and run later');
end

obd_file =strcat(folder,dir_info.name);
end

