function data = MinMaxData(data)
%--------------------------------------------------------------------------
% MINMAXDATA: this function is using min and max norm to normalize data
%
%   Function Signature:
%         data = MinMaxData(data)
%         @input:
%             data          :without norm data
%
%         @output:
%             data          :normalized data
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% @author: Xishuai Peng
% @email: xishuaip@umich.edu
% @date: May.30.2018
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%--------------------------------------------------------------------------
[row, col] = size(data);
min_v = min(data);
max_v = max(data);
min_v = repmat(min_v,row,1);
max_v = repmat(max_v,row,1);
data = (data - min_v)./(max_v - min_v + 5e-10);
end

