function mrcnnFeature(videoPath,outPath)
%--------------------------------------------------------------------------
% MRCNNFEATURE: this is the function to extract mask rcnn feature
%
%   Function Signature:
%         mrcnnFeature(videoPath,outPath)
%
%         @input:
%             videdoPath    :
%             outPath       :
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

bPath = pwd;
cd ('./Feature/')
modelpath = '../../../Model/mask_rcnn_coco.h5';
command = sprintf('activate & python TriWapper.py -video_inpath "%s"  -txt_outpath  "%s"  -modelPath "%s"& deactivate',videoPath,outPath);
system(command)
%py.importlib.import_module('D:\APP\Anaconda')
%pyversion D:\APP\Anaconda\\python.exe
% addpath ./
%py.importlib.import_module('skimage')
%py.TriWapper.loadMode()
%py.tss.pp(videoPath,outPath)
% commend = 'python TriWapper.py'
% system(commend)
cd (bPath)
end


