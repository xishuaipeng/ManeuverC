function targetIndex = findIndex(targetParams, OBDparams)
    OBDparams = split(OBDparams,'[');
    OBDparams = strtrim(OBDparams(:,:,1));
    OBDparams = strrep(OBDparams,' ','_');
    [~, targetIndex]  = ismember(targetParams, OBDparams);
end