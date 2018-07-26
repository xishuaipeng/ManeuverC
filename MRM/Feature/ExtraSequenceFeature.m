function seqFeature = ExtraSequenceFeature(data, featureField, videoPath,startIndex, lastIndex,out_path )
%--------------------------------------------------------------------------
% EXTRASEQUENCEFEATURE: this is the main function to extract sequence
% features according to featureField.
%
%   Function Signature:
%         seqFeature = ExtraSequenceFeature(data, featureField, videoPath,
%         startIndex, lastIndex, out_path)
%
%         @input:
%             data          : Class DataSet
%             featureField  : feature field, like 'LeftTurn', 'RightTurn', 'LeftTurnChange', 'RightTurnChange'
%             videoPath     : video data path
%             startIndex    : start index of sample data
%             lastIndex     : end index of sample data
%             out_path      : output path
%
%         @output:
%             seqFeature    :
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% @author: Xishuai Peng
% @email: xishuaip@umich.edu
% @date: May.30.2018
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%--------------------------------------------------------------------------
    tableField = data.Properties.VariableNames;
    %%%%
    maxIndex = size(data,1);
    maxSeq = size(startIndex,1);
    indexUnique=[];
    for i=1:maxSeq
        indexUnique = [indexUnique,startIndex(i):lastIndex(i)];
    end
    indexUnique = unique(indexUnique);
    seqFeature = table(startIndex,lastIndex,'VariableNames',{'startIndex';'lastIndex'});
    for i = 1: size(featureField,2)
        seqFeature= process(data, indexUnique, seqFeature,featureField{i},videoPath, out_path);
    end
    
    
    
    
    
    
    
%%%%%%%%%%%%%%%%%%%%
%     if sum(cellfun(@(x) strcmp({x},'VGGScene'),featureField)) > 0 & sum(cellfun(@(x) strcmp({x},'VGGScene'),tableField))== 0  
%         frameUnique = data.frame(indexUnique);
%         vgg_path = fullfile(out_path,'vggScene.mat');
%         vggFeature = generate_vgg_scene(vgg_path,videoPath, indexUnique, frameUnique);
%         seqFeature = feature2seq(vggFeature,seqFeature,'VGGScene');
%     end
    
%     if sum(cellfun(@(x) strcmp({x},'VGGObject'),featureField)) > 0 & sum(cellfun(@(x) strcmp({x},'VGGObject'),tableField))== 0  
%         frameUnique = data.frame(indexUnique);
%         vgg_path = fullfile(out_path,'vggObject.mat');
%         vggFeature = generate_vgg_object(vgg_path,videoPath, indexUnique, frameUnique);
%         seqFeature = feature2seq(vggFeature,seqFeature,'VGGObject');
%     end
    
%     if sum(cellfun(@(x) strcmp({x},'MIX'),featureField)) > 0 & sum(cellfun(@(x) strcmp({x},'MIX'),tableField))== 0  
%         frameUnique = data.frame(indexUnique);
%         vgg_path = fullfile(out_path,'vggObject.mat');
%         vggObject = generate_vgg_object(vgg_path,videoPath, indexUnique, frameUnique);
%         vgg_path = fullfile(out_path,'vggScene.mat');
%         vggScene = generate_vgg_scene(vgg_path,videoPath, indexUnique, frameUnique);
%         [A,B,r,V1,V2,stats] = canoncorr(vggObject(:,2:end),vggScene(:,2:end)) ;
%         nComponent = 400;
%         vggMix = zeros(size(vggObject,1), nComponent+1);
%         vggMix (:,1) = vggScene(:,1);
%         vggMix(:,2:end) = V1(:,1:nComponent)+V2(:,1:nComponent);
%         seqFeature = feature2seq(vggMix,seqFeature,'MIX');
%     end
    
      
%     if sum(cellfun(@(x) strcmp({x},'Heading'),featureField)) > 0 & sum(cellfun(@(x) strcmp({x},'Heading'),tableField))== 0  
% %         save_path = fullfile(out_path,'Heading.mat');
%         heading = data.GPS_heading(indexUnique);
%         heading = [indexUnique', heading];
%         seqFeature = feature2seq(heading,seqFeature,'Heading');
%         seqFeature.Heading = cellfun(@(x) headingPro(x), seqFeature.Heading,'UniformOutput',false);
%     end
    
%     if sum(cellfun(@(x) strcmp({x},'Speed'),featureField)) > 0 & sum(cellfun(@(x) strcmp({x},'Speed'),tableField))== 0  
%         Speed = data.speed(indexUnique);
%         Speed = [indexUnique', Speed];
%         seqFeature = feature2seq(Speed,seqFeature,'Speed');
%         seqFeature.Speed = cellfun(@(x) x/150, seqFeature.Speed,'UniformOutput',false);
%     end
    
%     if sum(cellfun(@(x) strcmp({x},'Curvature'),featureField)) > 0 & sum(cellfun(@(x) strcmp({x},'Curvature'),tableField))== 0  
%         x = data.GPS_lat(indexUnique);
%         y = data.GPS_long(indexUnique);
%         [x,y] = ll2utm(x,y);
%         Curvature = [indexUnique',[x,y]];
%         seqFeature = feature2seq(Curvature,seqFeature,'Curvature');
%         seqFeature.Curvature = cellfun(@(x) curvaturePro(x), seqFeature.Curvature,'UniformOutput',false);
%     end
%%%%%%%%%%%%%

        
    
end

function  seqFeature= process(data, indexUnique, seqFeature,featureName,videoPath, out_path)
    tableField = seqFeature.Properties.VariableNames;
    Frame = data.frame(indexUnique);
    Frame = [indexUnique', Frame];
    seqFeature = feature2seq(Frame,seqFeature,'Frame');
    if  sum(cellfun(@(x) strcmp({x},featureName),tableField))> 0  
        return;
    end
    if strcmp(featureName, 'Curvature') 
        x = data.GPS_lat(indexUnique);
        y = data.GPS_long(indexUnique);
        [x,y] = ll2utm(x,y);
        Curvature = [indexUnique',[x,y]];
        seqFeature = feature2seq(Curvature,seqFeature,'Curvature');
        seqFeature.Curvature = cellfun(@(x) curvaturePro(x), seqFeature.Curvature,'UniformOutput',false); 
    elseif strcmp(featureName, 'Speed') 
        Speed = data.speed(indexUnique);
        Speed = [indexUnique', Speed];
        seqFeature = feature2seq(Speed,seqFeature,'Speed');
        seqFeature.Speed = cellfun(@(x) x/150, seqFeature.Speed,'UniformOutput',false);
        
    elseif strcmp(featureName, 'Heading') 
%         save_path = fullfile(out_path,'Heading.mat');
        heading = data.GPS_heading(indexUnique);
        heading = [indexUnique', heading];
        seqFeature = feature2seq(heading,seqFeature,'Heading');
        seqFeature.Heading = cellfun(@(x) headingPro(x), seqFeature.Heading,'UniformOutput',false);
        
    elseif strcmp(featureName, 'VGGObject') 
        frameUnique = data.frame(indexUnique);
        vgg_path = fullfile(out_path,'vggObject.mat');
        vggFeature = generate_vgg_object(vgg_path,videoPath, indexUnique, frameUnique);
        seqFeature = feature2seq(vggFeature,seqFeature,'VGGObject');
        
    elseif strcmp(featureName, 'VGGScene') 
        frameUnique = data.frame(indexUnique);
        vgg_path = fullfile(out_path,'vggScene.mat');
        vggFeature = generate_vgg_scene(vgg_path,videoPath, indexUnique, frameUnique);
        seqFeature = feature2seq(vggFeature,seqFeature,'VGGScene');
    else
        try
            features = data.(featureName);
            features = features(indexUnique);
            features = [indexUnique', features];
            seqFeature = feature2seq(features,seqFeature,featureName);
        catch
            disp([featureName,' fail to extract!']);
        end
    end
        
        
        



end

function vggFeature = generate_vgg_object(feature_path, videoPath,indexUnique, frameIndex)
%--------------------------------------------------------------------------
% GENERATE_VGG_OBJECT: this function is to generate vgg object feature
%
%   Function Signature:
%         vggFeature = generate_vgg_object(feature_path, videoPath,indexUnique, frameIndex)
%--------------------------------------------------------------------------   
    if ~exist(feature_path)
                net = vgg19;
                n_case = length(frameIndex);
                videoObj = VideoReader(videoPath);
                vggFeature = zeros(n_case,4096+1);
                vggFeature(:,1) = indexUnique;
                for i = 1 : n_case
                    curFrame = frameIndex(i);
                    curImage = read(videoObj,curFrame);
                    vggFeature(i,2:end) = vgg19Feature(curImage,net);
                end
                save(feature_path, 'vggFeature');
     else
                clear vggFeature
                load(feature_path,'vggFeature')
    end
end

function vggFeature = generate_vgg_scene(feature_path, videoPath, indexUnique, frameIndex)
%--------------------------------------------------------------------------
% GENERATE_VGG_SCENE: this function is to generate vgg scene feature
%
%   Function Signature:
%         vggFeature = generate_vgg_scene(feature_path, videoPath,indexUnique, frameIndex)
%--------------------------------------------------------------------------  
    if ~exist(feature_path)
                curPath = mfilename('fullpath');
                file  = strsplit( curPath,["/","\"]);
                prototxtPath = fullfile(file{1:end-1}, 'VGG16', '../../../Model/deploy_vgg16_places365.prototxt');
                caffemodelPath = fullfile(file{1:end-1}, 'VGG16', '../../../Model/vgg16_places365.caffemodel');
                net = importCaffeNetwork(prototxtPath,caffemodelPath);
                n_case = length(frameIndex);
                videoObj = VideoReader(videoPath);
                vggFeature = zeros(n_case,4096+1);
                vggFeature(:,1) = indexUnique;
                for i = 1 : n_case
                    curFrame = frameIndex(i);
                    curImage = read(videoObj,curFrame);
                    curImage = im2double(curImage);
                    vggFeature(i,2:end) = vgg16Feature(curImage,net);
                end
                save(feature_path, 'vggFeature');
     else
                clear vggFeature
                load(feature_path,'vggFeature');
    end
end

function seqFeature = feature2seq(modelFeature,seqFeature,field)
%--------------------------------------------------------------------------
% FEATURE2SEQ: this funtion is to transform specific event feature to
% sequence feature and add to seqFeature.
%
%   Function Signature:
%         seqFeature = feature2seq(modelFeature,seqFeature,field)
%--------------------------------------------------------------------------  
        modelIndex = modelFeature(:,1);
        maxSeq = size(seqFeature,1);
        for i = 1:maxSeq 
            seq_start = seqFeature.startIndex(i);
            seq_duration = seqFeature.lastIndex(i)- seq_start;
            index = find( modelIndex == seqFeature.startIndex(i));
            if(length( modelIndex( index + seq_duration) == seqFeature.lastIndex(i))==0)
                fprintf('index is not matched in %s mat!',field );
                return;
            end
        seqFeature.(field){i}= modelFeature(index : index + seq_duration, 2:end);
        end
end

function headFeature = headingPro(x)
%--------------------------------------------------------------------------
% HEADINGPRO: this function is to process heading and normalize heading
% feature
%
%   Function Signature:
%         headFeature = headingPro(x)
%-------------------------------------------------------------------------- 
    cosx = cos(x*pi/180);%[-1,1]
    meanHeading = mean(cosx);%[-1,1]
    cosx = cosx - meanHeading;%[-2,2]
    cosx = abs(cosx/2);%[0,1]
    headFeature = cosx;
end

function curvatureFeature = curvaturePro(data)
%--------------------------------------------------------------------------
% CURVATUREPRO: this function is to generate curvature from GPS
%
%   Function Signature:
%         curvatureFeature = curvaturePro(data)
%-------------------------------------------------------------------------- 
    x = data(:,1);
    y = data(:,2);
    t = [1:1:length(x)]';
    f = polyfit(t,x,3);    
    g = polyfit(t,y,3);
%     
    f_1 = polyder(f);
    g_1 = polyder(g);
    f_2 = polyder(f_1);
    g_2 = polyder(g_1);
    
    x_1 = polyval(f_1, t);
    x_2 = polyval(f_2,t);
    y_1 = polyval(g_1, t);
    y_2 = polyval(g_2, t);
    curvatureFeature =  abs( x_1 .* y_2 - y_1 .* x_2) ./ (x_1 .^2 + y_1.^2 ).^(3/2);
    curvatureFeature(isnan(curvatureFeature))=5000;
    curvatureFeature(curvatureFeature>5000)=5000;
    curvatureFeature=curvatureFeature/5000;

end

function angleList = angRangeChange(angleList, varargin)
%--------------------------------------------------------------------------
% ANGRANGECHANGE: this function is to change angle range
% angleList = angRangeChange(angleList, outMax)
% angleList = angRangeChange(angleList)
%--------------------------------------------------------------------------
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
 
end

end % EOF

