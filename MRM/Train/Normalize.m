function data = Normalize(data)
%--------------------------------------------------------------------------
% NORMALIZE: this function is to normalize different feature data
%
%   Function Signature:
%         data = Normalize(data)
%         @input:
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
            mean_v = mean(data,1);
            var_v = var(data,[],1);
            mean_v = repmat(mean_v,row,1);
            var_v = repmat(var_v,row,1);
            data = (data - mean_v)./((var_v.^2+ 1e-10).^(0.5));
end
