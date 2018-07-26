function sigData = extractSigCSVdata(fid, paramsVector)
    dataFormat = repmat('%s', 1, size(paramsVector, 2));
%     testLine = fgetl(fid);
%     testData = textscan(testLine, dataFormat, 'delimiter', ',');
%     
%     error = 0;
%     for e = 1:size(paramsVector, 2) 
%         if size(testData{1, e}, 1) > 1
%             error = error + 1;
%         end
%     end
  
    dataFormat = [dataFormat, repmat('%s', 1, 1)];
    temp = textscan(fid, dataFormat, 'delimiter', ',');
    dataLen = length(temp{1,1});
    for i = 2:size(paramsVector,2)
        dataLen = min(dataLen, length(temp{1, i}));
    end
    sigData = cell(dataLen, size(temp, 2));
    for i=1:size(paramsVector, 2)
        sigData(1:dataLen, i) = temp{1, i}(1:dataLen);
    end
end