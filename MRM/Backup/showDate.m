function   showDate(data)
%--------------------------------------------------------------------------
% SHOWDATE: this is the function to show heading, speed and video data for
% demo using class Dataset.
%
%   Function Signature:
%         showDate(data)
%
%         @input:
%             data          :Class Dataset
%
%         @output:
%             None          :
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% @author: Xishuai Peng
% @email: xishuaip@umich.edu
% @date: May.30.2018
% @copyright: Intelligent System Laboratory University of Michigan-Dearborn
%--------------------------------------------------------------------------
tr = imread('./sign/TR.jpg');
tl = imread('./sign/TL.jpg');
llc = imread('./sign/LLC.jpg');
rlc = imread('./sign/RLC.jpg');
gs = imread('./sign/GS.jpg');
videoPath = sprintf('./input/raw_data/%s/%s_video.avi',data.dataID,data.dataID);%avi
vidObj =  VideoReader(videoPath);
FrameRate = vidObj.FrameRate;
trainNum= length(data.X);
figure;hold on;
for i = 1: trainNum
   [numCase,c] = size(data.trainingFrame{i});
   pause(3);
   i
%    frameBegin = data.trainingFrame{i}(1); 
%    frameLast= data.trainingFrame{i}(end);
   subplot(4,4,16);
   switch data.Y(i)
    case 1
        imshow(tl)
    case 2
        imshow(tr)
    case 3
        imshow(llc)
    case 4
        imshow(rlc)
    otherwise
        imshow(gs)
   end
   for j =1:numCase
        img = read(vidObj, data.trainingFrame{i}(j) );
        subplot(4,4,[1,2,3,5,6,7,9,10,11,13,14,15]);imshow(img,[]);
        subplot(4,4,12);;title('Heading');plot([1,j],[data.trainingData{i}(1,2),data.trainingData{i}(j,2)],'-');
        subplot(4,4,8);title('Speed');plot([1,j],[data.trainingData{i}(1,3),data.trainingData{i}(j,3)],'-');
        drawnow();
    
   end

end
end

