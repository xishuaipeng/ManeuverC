function [OBDdata,nonzero_ind] = getOBDdata(targetParams, OBDparams, fid)
    targetIndex = findIndex(targetParams, OBDparams);
    zero_ind = find(targetIndex == 0);
    if ~all(targetIndex)        
        warning([targetParams{zero_ind} ' doesnot exist']);
    end
%     [~, targetIndex]  = ismember(targetParams, OBDparams);
    OBDallData = extractSigCSVdata(fid, OBDparams);
    nonzero_ind = find(targetIndex ~= 0);
    targetIndex(zero_ind) = [];
    OBDdata = OBDallData(:, targetIndex);
end
