%% Process the Brats15 data into png file
clc
clear
im_sz = 256;
%%
Flair_dir = 'D:\DATA\Brats2018\Ver_2\Train\NII\F';
dir_file = Flair_dir;
list_file = dir(dir_file);
for k =1:length(list_file)-2
   file_name = fullfile(list_file(k+2).folder, list_file(k+2).name);
   Img_all = niftiread(file_name);
   img = double(Img_all(:,:,45:end-35));
   resize_img = zeros(im_sz,im_sz,size(img,3));
   for l = 1:size(img,3)
      resize_img(:,:,l) = imresize(img(:,:,l),[im_sz,im_sz]); 
   end
   u = double(mean(resize_img(:)));
   v = double(std(resize_img(:)));
   norm_img =  (resize_img-u)./(v);
   norm_img(norm_img>5.3) = 5.3;
   norm_img(norm_img<-0.5) = -0.5;
   norm_img = (norm_img-0.5)./(4.8);
   norm_img = permute(norm_img,[3,1,2]);
   
   for j = 1:size(norm_img,1)
       img = flipud(rot90(squeeze(norm_img(j,:,:)))); 
       imwrite(img,['D:\DATA\Brats2018\Ver_2\Train\IMG\F\', num2str(k), '_',num2str(j), '.png']);
   end
end

%%
T1_dir = 'D:\DATA\Brats2018\Ver_2\Train\NII\T1';
dir_file = T1_dir;
list_file = dir(dir_file);
for k =1:length(list_file)-2
   file_name = fullfile(list_file(k+2).folder, list_file(k+2).name);
   Img_all = niftiread(file_name);
   img = double(Img_all(:,:,45:end-35));
   resize_img = zeros(im_sz,im_sz,size(img,3));
   for l = 1:size(img,3)
      resize_img(:,:,l) = imresize(img(:,:,l),[im_sz,im_sz]); 
   end
   u = double(mean(resize_img(:)));
   v = double(std(resize_img(:)));
   norm_img =  (resize_img-u)./(v);
   norm_img(norm_img>5.3) = 5.3;
   norm_img(norm_img<-0.5) = -0.5;
   norm_img = (norm_img-0.5)./(4.8);
   norm_img = permute(norm_img,[3,1,2]);
   
   for j = 1:size(norm_img,1)
       img = flipud(rot90(squeeze(norm_img(j,:,:)))); 
       imwrite(img,['D:\DATA\Brats2018\Ver_2\Train\IMG\T1\', num2str(k), '_',num2str(j), '.png']);
   end
end
%%
T1c_dir = 'D:\DATA\Brats2018\Ver_2\Train\NII\T1c';
dir_file = T1c_dir;
list_file = dir(dir_file);
for k =1:length(list_file)-2
   file_name = fullfile(list_file(k+2).folder, list_file(k+2).name);
   Img_all = niftiread(file_name);
   img = double(Img_all(:,:,45:end-35));
   resize_img = zeros(im_sz,im_sz,size(img,3));
   for l = 1:size(img,3)
      resize_img(:,:,l) = imresize(img(:,:,l),[im_sz,im_sz]); 
   end
   u = double(mean(resize_img(:)));
   v = double(std(resize_img(:)));
   norm_img =  (resize_img-u)./(v);
   norm_img(norm_img>5.3) = 5.3;
   norm_img(norm_img<-0.5) = -0.5;
   norm_img = (norm_img-0.5)./(4.8);
   norm_img = permute(norm_img,[3,1,2]);
   
   for j = 1:size(norm_img,1)
       img = flipud(rot90(squeeze(norm_img(j,:,:)))); 
       imwrite(img,['D:\DATA\Brats2018\Ver_2\Train\IMG\T1c\', num2str(k), '_',num2str(j), '.png']);
   end
end
%%
T2_dir = 'D:\DATA\Brats2018\Ver_2\Train\NII\T2';
dir_file = T2_dir;
list_file = dir(dir_file);
for k =1:length(list_file)-2
   file_name = fullfile(list_file(k+2).folder, list_file(k+2).name);
   Img_all = niftiread(file_name);
   img = double(Img_all(:,:,45:end-35));
   resize_img = zeros(im_sz,im_sz,size(img,3));
   for l = 1:size(img,3)
      resize_img(:,:,l) = imresize(img(:,:,l),[im_sz,im_sz]); 
   end
   u = double(mean(resize_img(:)));
   v = double(std(resize_img(:)));
   norm_img =  (resize_img-u)./(v);
   norm_img(norm_img>5.3) = 5.3;
   norm_img(norm_img<-0.5) = -0.5;
   norm_img = (norm_img-0.5)./(4.8);
   norm_img = permute(norm_img,[3,1,2]);
   
   for j = 1:size(norm_img,1)
       img = flipud(rot90(squeeze(norm_img(j,:,:)))); 
       imwrite(img,['D:\DATA\Brats2018\Ver_2\Train\IMG\T2\', num2str(k), '_',num2str(j), '.png']);
   end
end
%%
M_dir = 'D:\DATA\Brats2018\Ver_2\Train\M\NII';
dir_file = M_dir;
list_file = dir(dir_file);
mask = zeros(149*76, im_sz, im_sz);
for k =1:length(list_file)-2
   file_name = fullfile(list_file(k+2).folder, list_file(k+2).name);
   Img_all = niftiread(file_name);
   img = double(Img_all(:,:,45:end-35));
   resize_img = zeros(im_sz,im_sz,size(img,3));
   for l = 1:size(img,3)
      resize_img(:,:,l) = imresize(img(:,:,l),[im_sz,im_sz]); 
   end
   norm_img = resize_img./4;   
   norm_img = permute(norm_img,[3,1,2]);
   mask(76*(k-1)+1:76*k, :, :) = norm_img;
%    for j = 1:size(norm_img,1)
%        img = flipud(rot90(squeeze(norm_img(j,:,:)))); 
%        imwrite(img,['D:\DATA\Brats2018\Ver_2\Train\IMG\M\', num2str(k), '_',num2str(j), '.png']);
%    end
end

%%
Flair_dir = 'D:\DATA\Brats2018\Ver_2\Test\NII\F';
dir_file = Flair_dir;
list_file = dir(dir_file);
for k =1:length(list_file)-2
   file_name = fullfile(list_file(k+2).folder, list_file(k+2).name);
   Img_all = niftiread(file_name);
   img = double(Img_all(:,:,45:end-35));
   resize_img = zeros(im_sz,im_sz,size(img,3));
   for l = 1:size(img,3)
      resize_img(:,:,l) = imresize(img(:,:,l),[im_sz,im_sz]); 
   end
   u = double(mean(resize_img(:)));
   v = double(std(resize_img(:)));
   norm_img =  (resize_img-u)./(v);
   norm_img(norm_img>5.3) = 5.3;
   norm_img(norm_img<-0.5) = -0.5;
   norm_img = (norm_img-0.5)./(4.8);
   norm_img = permute(norm_img,[3,1,2]);
   
   for j = 1:size(norm_img,1)
       img = flipud(rot90(squeeze(norm_img(j,:,:)))); 
       imwrite(img,['D:\DATA\Brats2018\Ver_2\Test\IMG\F\', num2str(k), '_',num2str(j), '.png']);
   end
end
%%
T1_dir = 'D:\DATA\Brats2018\Ver_2\Test\NII\T1';
dir_file = T1_dir;
list_file = dir(dir_file);
for k =1:length(list_file)-2
   file_name = fullfile(list_file(k+2).folder, list_file(k+2).name);
   Img_all = niftiread(file_name);
   img = double(Img_all(:,:,45:end-35));
   resize_img = zeros(im_sz,im_sz,size(img,3));
   for l = 1:size(img,3)
      resize_img(:,:,l) = imresize(img(:,:,l),[im_sz,im_sz]); 
   end
   u = double(mean(resize_img(:)));
   v = double(std(resize_img(:)));
   norm_img =  (resize_img-u)./(v);
   norm_img(norm_img>5.3) = 5.3;
   norm_img(norm_img<-0.5) = -0.5;
   norm_img = (norm_img-0.5)./(4.8);
   norm_img = permute(norm_img,[3,1,2]);
   
   for j = 1:size(norm_img,1)
       img = flipud(rot90(squeeze(norm_img(j,:,:)))); 
       imwrite(img,['D:\DATA\Brats2018\Ver_2\Test\IMG\T1\', num2str(k), '_',num2str(j), '.png']);
   end
end

%%
T1c_dir = 'D:\DATA\Brats2018\Ver_2\Test\NII\T1c';
dir_file = T1c_dir;
list_file = dir(dir_file);
for k =1:length(list_file)-2
   file_name = fullfile(list_file(k+2).folder, list_file(k+2).name);
   Img_all = niftiread(file_name);
   img = double(Img_all(:,:,45:end-35));
   resize_img = zeros(im_sz,im_sz,size(img,3));
   for l = 1:size(img,3)
      resize_img(:,:,l) = imresize(img(:,:,l),[im_sz,im_sz]); 
   end
   u = double(mean(resize_img(:)));
   v = double(std(resize_img(:)));
   norm_img =  (resize_img-u)./(v);
   norm_img(norm_img>5.3) = 5.3;
   norm_img(norm_img<-0.5) = -0.5;
   norm_img = (norm_img-0.5)./(4.8);
   norm_img = permute(norm_img,[3,1,2]);
   
   for j = 1:size(norm_img,1)
       img = flipud(rot90(squeeze(norm_img(j,:,:)))); 
       imwrite(img,['D:\DATA\Brats2018\Ver_2\Test\IMG\T1c\', num2str(k), '_',num2str(j), '.png']);
   end
end

%%
T2_dir = 'D:\DATA\Brats2018\Ver_2\Test\NII\T2';
dir_file = T2_dir;
list_file = dir(dir_file);
for k =1:length(list_file)-2
   file_name = fullfile(list_file(k+2).folder, list_file(k+2).name);
   Img_all = niftiread(file_name);
   img = double(Img_all(:,:,45:end-35));
   resize_img = zeros(im_sz,im_sz,size(img,3));
   for l = 1:size(img,3)
      resize_img(:,:,l) = imresize(img(:,:,l),[im_sz,im_sz]); 
   end
   u = double(mean(resize_img(:)));
   v = double(std(resize_img(:)));
   norm_img =  (resize_img-u)./(v);
   norm_img(norm_img>5.3) = 5.3;
   norm_img(norm_img<-0.5) = -0.5;
   norm_img = (norm_img-0.5)./(4.8);
   norm_img = permute(norm_img,[3,1,2]);
   
   for j = 1:size(norm_img,1)
       img = flipud(rot90(squeeze(norm_img(j,:,:)))); 
       imwrite(img,['D:\DATA\Brats2018\Ver_2\Test\IMG\T2\', num2str(k), '_',num2str(j), '.png']);
   end
end

%%
M_dir = 'D:\DATA\Brats2018\Ver_2\Test\NII\M';
dir_file = M_dir;
list_file = dir(dir_file);
for k =1:length(list_file)-2
   file_name = fullfile(list_file(k+2).folder, list_file(k+2).name);
   Img_all = niftiread(file_name);
   img = double(Img_all(:,:,45:end-35));
   resize_img = zeros(im_sz,im_sz,size(img,3));
   for l = 1:size(img,3)
      resize_img(:,:,l) = imresize(img(:,:,l),[im_sz,im_sz]); 
   end
   norm_img = resize_img./4;   
   norm_img = permute(norm_img,[3,1,2]);
   
   for j = 1:size(norm_img,1)
       img = flipud(rot90(squeeze(norm_img(j,:,:)))); 
       imwrite(img,['D:\DATA\Brats2018\Ver_2\Test\IMG\M\', num2str(k), '_',num2str(j), '.png']);
   end
end

