function checkCurvature(x,y,k)
%--------------------------------------------------------------------------
% CHECKCURVATURE: this is the function to check curvature using scatter
%
%   Function Signature:
%         checkCurvature(x,y,k)
%
%         @input:
%             x             :latitude
%             y             :longtitude
%             k             :label result
%
%         @output:
%             None
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% @author: Xishuai Peng
% @email: xishuaip@umich.edu
% @date: May.30.2018
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%--------------------------------------------------------------------------
      maxLength = length(x);
     % k = smooth(k,20);
      label = k;
      classes = 2;
      label(label<700)=1;
      label(label>700)=2;
      %label(label<1)=3;
      
      radio = jet( classes);
      figure; hold on;
      for i = 1: classes
          index = find(label == i);
          p_x = x(index);
          p_y = y(index);
          color = radio(i,:);
          scatter(p_x,p_y, 'MarkerFaceColor',color,'MarkerEdgeColor',color);  
      end

end