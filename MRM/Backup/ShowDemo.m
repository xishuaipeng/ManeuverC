function  ShowDemo(dataID,logField,eventField,trainingFrame,Y ) 
%--------------------------------------------------------------------------
% SHOWDATE: this is the function to show heading, speed and video data for
% demo without using class Dataset.
%
%   Function Signature:
%         ShowDemo(dataID,logField,eventField,trainingFrame,Y)
%
%         @input:
%             dataID        : data ID, IDXXX_TXXX
%             logField      : datalog field
%             eventField    : event field
%             trainingFrame :
%             Y             :
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
data = Dataset(dataID, logField, eventField);
videoObj = VideoReader(data.videoPath);
data = data.readLogdata();
data = data.reSync('./input/sycnFile.txt');           
vgg_net = vgg19;
fig = figure;hold on;
set (gcf,'Position',[1,1,1920,1080], 'color','w')
for i = 164:166 %1: length(trainingFrame)

    minF = min(trainingFrame{i});
    maxF = max(trainingFrame{i});
    dirName = ['./output/' data.dataID '/' num2str(i)];
    mkdir(dirName);
    timeTable= ([minF-50:maxF]./data.frameRate + data.timeDelayforVideo);
    timeIndex = round(timeTable./data.logRate);
    timeIndex(timeIndex<1)=1;
    timeIndex(timeIndex>size(data.logData,1)) = size(data.logData,1);
    speed = data.logData.speed(timeIndex);
    heading = data.logData.GPS_heading(timeIndex);  
       %label
        subplot(4,4,16);
        switch Y(i)
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
        title('event');
    for j = minF:1: maxF
        frame = read(videoObj, j);
        subplot(4,4,[1,2,3,5,6,7,9,10,11,13,14,15]);
        %subtitle = sprintf('Front View video, frame index: %d',j);
        subtitle = 'Front View video';
        imshow(frame, 'Border', 'tight'); title(subtitle);
        %feature map
         featuremap = activations(vgg_net,frame,'conv5_4','OutputAs','channels');
         [maxValue,maxValueIndex] = max(max(max(featuremap)));
         featuremap = featuremap(:,:,maxValueIndex);
        
         featuremap = imresize(featuremap,[360 540]);
         subplot(4,4,4);imshow(featuremap,[]);title('VGG 19 Feature Map');
        % %speed
        begin = j-minF + 1;
        showIndex= [begin : begin+50];
        showspeed =  speed(showIndex);
        subplot(4,4,8); plot(timeTable(showIndex), ...
        showspeed,'-');title('Speed');
        ylim([0 200]);  
        xlim([timeTable(begin) timeTable(begin+50)]);
        
        
        xlabel('sec'); ylabel('mph');
        %heading
        showheading =  heading(showIndex);
        subplot(4,4,12); plot(timeTable(showIndex), ...
            showheading,'-');title('Heading');
        ylim([0 360]);  
        xlim([timeTable(begin) timeTable(begin+50)]);
        xlabel('sec'); ylabel('degree'); 
        %drawnow();
        outputName = [dirName '/' num2str(j)];
        print(fig, outputName, '-djpeg');
    end
        prompt = 'Keep it? ';
        x = input(prompt);
        if x==0
            rmdir(dirName,'s');
        end
end





end

