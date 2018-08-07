clear; close all;
%% settings
folder = './Train/Set14/';%
out_dir = './Train/';
size_input = 8;% There are 4 pixels padding. Paper presents 7
size_label = 56;% (11-4) *3 - (3-1)
scale = 4;
stride = 56;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;
cur = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));

for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder, filepaths(i).name));
    if size(image, 3) > 1
        image = rgb2ycbcr(image);
        image = im2double(image(:, :, 1));
    end
    cur = cur + 1;
    disp([num2str(cur) '/' num2str(length(filepaths))]);
    im_label = modcrop(image, scale);
    im_input = imresize(im_label, 1/scale, 'bicubic');
    im_input_bic = imresize(im_input, scale, 'bicubic');
    hei = size(im_label,1);
    wid = size(im_label,2);
    
    for x = 1 : stride : hei - size_label - 20
        for y = 1 : stride : wid - size_label - 20
            
            subim_label = im_label(x : size_label + x - 1, y : size_label + y - 1,:);
            subim_input = imresize(subim_label, 1/scale, 'bicubic');
            count = count + 1;
            imwrite(subim_label, [out_dir 'Patch/test/' num2str(count) '.png']);
        end
    end
end