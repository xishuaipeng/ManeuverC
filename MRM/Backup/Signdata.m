classdef  Signdata
%--------------------------------------------------------------------------
% Dataset: this is the class to process video and OBD data, and extract
% features
%
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% @author: Xishuai Peng
% @email: xishuaip@umich.edu
% @date: May.30.2018
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%--------------------------------------------------------------------------
properties
baseData;
X;
Y;
 end
 methods
     function obj = Signdata(dataID)
        %--------------------------------------------------------------
        % MANEUVERDATA: this function is a constructor, adding and
        % initializing the class local parameters.
        %
        %   Function Signature:
        %         obj = Maneuverdata(varargin)
        %--------------------------------------------------------------         
         feature_field = {'time','speed','GPS_long','GPS_lat','GPS_heading','distance'};
         event_field = {'Stop','TrafficLight'};
         obj.baseData = Dataset(dataID,feature_field, event_field);
         obj = obj.data();
         obj.Y ={};
         obj.X ={};
     end
     
     function obj = data(varargin)
         obj = varargin{1};
         obj.baseData = obj.baseData.readLogdata();
         obj.baseData = obj.baseData.resampLogdata('distance',0.002);
         obj.baseData =  obj.baseData.labelSampledata();
         %obj.baseData = obj.baseData.segtrip(20,10,0.5,'time');
     end
     function obj = trainData(varargin)
         obj = varargin{1};
         event = obj.baseData.eventLabel{:,obj.baseData.eventField};
         index = [];
         for i=1: size(obj.baseData.eventField, 2)
            index = horzcat(index, find(event(:,i)==1)');
         end
         cellindex=1;
         for i=1:length(index)
            startEvent= round(timeParser(obj.baseData.eventLabel.StartTime{index(i)})* obj.baseData.frameRate);
            endEvent =  round(timeParser(obj.baseData.eventLabel.EndTime{index(i)})* obj.baseData.frameRate);
            for j = 1:  size(obj.baseData.segData,2)
                startFrame = obj.baseData.segData(j).minFrame;
                endFrame = obj.baseData.segData(j).maxFrame;
                start = max(startFrame, startEvent);
                last = min(endEvent,endFrame);
                duration = last - start;
                radio_event = duration /(endEvent- startEvent);
				radio_seg  =  duration /(endFrame - startFrame);
                radio = max(radio_event,radio_seg);
                if radio > 0.5
                    obj.X(cellindex) = {obj.baseData.frameLabel{startFrame:endFrame,:}}
                    obj.Y(cellindex) =  {obj.baseData.frameLabel{startFrame:endFrame,:}};
                    cellindex = cellindex + 1;
                end  
            end 
         end

         
         
     end
     
     
     
 end
    
    
    
end
