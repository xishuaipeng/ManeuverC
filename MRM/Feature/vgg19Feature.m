function [feature] = vgg19Feature(image,net)
if isempty(net)
    net = vgg19;
end
sz = net.Layers(1).InputSize;
image = imresize(image,[sz(1),sz(2)]);
feature = activations(net,image,43);
end

