clear;
%%

camera = webcam('Kinect V2 Video Sensor');
load('NetStreamExp.mat');
inputSize = net.Layers(1).InputSize(1:2)

%%
h = figure;
h.Position(3) = 2*h.Position(3);
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
% In the left subplot, display the image and classification together.
im = snapshot(camera);
image(ax1,im)
im = imresize(im,inputSize);
[label,score] = classify(net,im);
title(ax1,{char(label),num2str(max(score),2)});
% Select the top five predictions by selecting the classes with the highest scores.
[~,idx] = sort(score,'descend');
idx = idx(5:-1:1);
classes = net.Layers(end).Classes;
classNamesTop = string(classes(idx));
scoreTop = score(idx);

%%
% h = figure;
h.Position(3) = 2*h.Position(3);
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
% ax2.PositionConstraint = 'innerposition';
% Continuously display and classify images
% together with a histogram of the top five predictions.
while ishandle(h)
    % Display and classify the image
    im = snapshot(camera);
    image(ax1,im)
    im = imresize(im,inputSize);
    [label,score] = classify(net,im);
    title(ax1,{char(label),num2str(max(score),2)},'FontSize',20,'FontWeight','bold','Color','r');
    % Select the top five predictions
    [~,idx] = sort(score,'descend');
    idx = idx(5:-1:1);
    scoreTop = score(idx);
    classNamesTop = string(classes(idx));
    % Plot the histogram
    barh(ax2,scoreTop)
    title(ax2,'Facial Expressions')
    xlabel(ax2,'Probability','FontSize',17)
    xlim(ax2,[0 1])
    yticklabels(ax2,classNamesTop)
    ax2.YAxisLocation = 'right';
    drawnow
end
%%
% In order to close the Kinect sensor
% clear('camera');



