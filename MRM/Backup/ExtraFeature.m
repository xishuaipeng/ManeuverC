function data = ExtraFeature(data, featureField, videoPath )
%--------------------------------------------------------------------------
% EXTRAFEATURE: this is the function to extract event
% features according to featureField using class Dataset.
%
%   Function Signature:
%         data = ExtraFeature(data, featureField, videoPath )
%
%         @input:
%             data          : class Dataset
%             featureField  : feature field, like 'LeftTurn', 'RightTurn', 'LeftTurnChange', 'RightTurnChange'
%             videoPath     : video data path
%
%         @output:
%             data          :
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% @author: Xishuai Peng
% @email: xishuaip@umich.edu
% @date: May.30.2018
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%--------------------------------------------------------------------------
    tableField = data.Properties.VariableNames;
    maxIndex = size(data,1);
    if sum(cellfun(@(x) strcmp({x},'VGG'),featureField))>0 
        if sum(cellfun(@(x) strcmp({x},'VGG'),tableField)) == 0
              net = vgg19;
%             curPath = mfilename('fullpath');
%             file  = strsplit( curPath,["/","\"]);
%             feature =0;
%             prototxtPath = fullfile(file{1:end-1}, 'VGG16', 'deploy_vgg16_places365.prototxt');
%             caffemodelPath = fullfile(file{1:end-1}, 'VGG16', 'vgg16_places365.caffemodel');
%             net = importCaffeNetwork(prototxtPath,caffemodelPath);
    
             videoObj = VideoReader(videoPath);
             %frameIndex = data.frame - data.frame(0) + 1;
             %maxFrame = length(frameIndex);
             vggFeature = zeros(maxIndex,4096);
             for i = 1 : maxIndex
                %curFrame = frameIndex(i);
                curImage = read(videoObj,i);
                vggFeature(i,:) = vgg19Feature(curImage,net);
             end
        end
         data.VGG =  vggFeature;
         
    end
    
    if sum(cellfun(@(x) strcmp({x},'Curvature'),featureField))>0 
       if sum(cellfun(@(x) strcmp({x},'Curvature'),tableField)) == 0
			[x,y] = ll2utm(data.GPS_lat,data.GPS_long);
            kappa = Curvature(x,y);
            kappa(kappa >2000) = 2000;
            kappa(isnan(kappa)) = 2000;
            kappa = kappa/2000;
            kappa = smooth(kappa,0.2,'rloess');
            %feature(:,index) = kappa;
            data.Curvature =  kappa;
       end
    end
    if sum(cellfun(@(x) strcmp({x},'Heading'),featureField))>0 
       if sum(cellfun(@(x) strcmp({x},'Heading'),tableField))==0
           smoothHeading = smooth(data.GPS_heading, 0.2, 'rloess');
           cosHeading = cos(pi * smoothHeading/180);
           cosHeading = (cosHeading+1)/2;
%            gradHeading = smoothHeading(2:end) - smoothHeading(1:end-1);
%            if any(gradHeading > 180)
%                 smoothHeading = angRangeChange(smoothHeading);
%            end
%             meanH =   mean(smoothHeading);
%            tHeading = smoothHeading - meanH ;
           data.Heading =  cosHeading ;
       end
    end
    if sum(cellfun(@(x) strcmp({x},'Speed'),featureField))>0
        if sum(cellfun(@(x) strcmp({x},'Speed'),tableField))==0
            speeding = data.speed/160;
            data.Speed = speeding;
        end
         %feature(:,index) = speeding;
    end
    if sum(cellfun(@(x) strcmp({x},'Mask'),featureField))>0 
       if  sum(cellfun(@(x) strcmp({x},'Mask'),tableField))==0
            outPath = replace(videoPath,'.avi','_mask.txt');
            outPath = [pwd,outPath(2:end)];
            if ~isfile(outPath)
                tempVideo = [pwd,videoPath(2:end)];
                mrcnnFeature(tempVideo,outPath);
            end       
            fid = fopen(outPath);
            maskFeature = [];
            while ~feof(fid)
                tline = fgetl(fid);
                if isempty( tline)
                    continue;
                end
                tline = split(tline,',');
                maskFeature = [maskFeature, tline];
            end
            fclose(fid);
            maskFeature = cellfun(@(x) str2num(x),maskFeature);
            maskFeature = maskFeature(2:end,:)';
            %maskFeature = maskFeature(:,frameIndex)';
            data.Mask = maskFeature;
       end
    end
    
end



function angleList = angRangeChange(angleList, varargin)
% change angle range
% angleList = angRangeChange(angleList, outMax)
% angleList = angRangeChange(angleList)
% written by Ruirui Liu
if nargin == 2
    outMax = varargin{1};
for i = 1: length(angleList)
    if angleList(i)<outMax-360
        n = ceil((outMax-360-angleList(i))/360);
        angleList(i) = angleList(i)+n*360;
    elseif angleList(i)>outMax
        n = ceil((angleList(i)-outMax)/360);
        angleList(i) = angleList(i)-n*360;
    end
end
elseif nargin == 1
    for i = 2:length(angleList)
        if abs(angleList(i) - angleList(i-1)) > 180
            angleList(i) = angleList(i) - ...
                360*round((angleList(i) - angleList(i-1))/360);
        end
%         theta1 = angleList(i-1)/180*pi;
%         theta2 = angleList(i)/180*pi;
%         angleList(i) = angleList(i-1) + ...
%             asin(sin(theta2-theta1))/pi*180;
    end
else 
end

end % EOF

