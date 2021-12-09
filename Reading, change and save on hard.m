clc;
clear;;
path='IKFDB';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
fsize=filesnumber(1,1);
for i = 1 : fsize
images{i} = imread(fullfile(path,fileinfo(i).name));
    disp(['Loading image No :   ' num2str(i) ]);
end;
% Adjust
% for i = 1:fsize   
% sizee{i} = imadjust(images{i});
%     disp(['Adjusting intensity value :   ' num2str(i) ]);
% end;
% Histogram eq
% for i = 1:fsize   
% sizee{i} = histeq(images{i});
%     disp(['Histogram eq :   ' num2str(i) ]);
% end;


% for i = 1:fsize   
% sizee{i} = cat(3,images{i},images{i},images{i});
% end;

% % Resize
for i = 1:fsize   
sizee{i} = imresize(images{i},[64 64]);
end;
% 
% for i = 1:fsize   
% sizee{i} = uint8(sizee{i});
% end;

%save to disk
for i = 1:fsize   
   imwrite(sizee{i},strcat('my_new',num2str(i),'.jpg'));
end