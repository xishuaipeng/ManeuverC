function time = timeParser(date)
%--------------------------------------------------------------------------
% EXTRASEQUENCEFEATURE: tranfer the time from 'MM:SS.FFF' or
% 'HH:MM:SS.FFF' to seconds.
%
%   Function Signature:
%         time = timeParser(date)
%
%         @input:
%             date          : 'MM:SS.FFF' or 'HH:MM:SS.FFF'
%            
%         @output:
%             time          : time in seconds
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% @author: Xishuai Peng
% @email: xishuaip@umich.edu
% @date: May.30.2018
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%--------------------------------------------------------------------------
     % tranfer the time from 'MM:SS.FFF' or 'HH:MM:SS.FFF' to seconds
         colonIndex = find(date == ':');
         if length(colonIndex)==1
             dateVector = datevec(date, 'MM:SS.FFF');
         elseif length(colonIndex)==2
             dateVector = datevec(date, 'HH:MM:SS.FFF');
         else
                'Time can not formated to second'
                time = date
                return 
         end
         time =  seconds(duration(dateVector(:,4),dateVector(:,5), dateVector(:,6)));     
end

