clc;
clear;

image_path = './test/X/';
filenames = dir(image_path);  % structure[name, data, bytes, isdir, datenum]
image_filename = filenames(3:end);
filter = fspecial('gaussian', [8 8], 2);

for i=1:20
     fprintf('Testing image %d...\n', i);
     file_name = image_filename(i).name;
     data = load([image_path, file_name]);
     data = data.msi;
%      LR = imfilter(data,filter);
     LR = imresize(data, 0.125, 'nearest');
     save(['./test/Z_clearn/', file_name], 'LR');
     pause(1);
end