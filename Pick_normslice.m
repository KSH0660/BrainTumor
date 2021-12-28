clc
clear
im_sz = 256;
%%
M_dir = 'D:\DATA\Brats2018\Ver_2\Train\M\5';
M_save_dir = 'D:\DATA\Brats2018\Ver_2\Norm_slice\M';
M_files = dir(M_dir); 

F_dir = 'D:\DATA\Brats2018\Ver_2\Train\F\5';
F_save_dir = 'D:\DATA\Brats2018\Ver_2\Norm_slice\F';
F_files = dir(F_dir); 

T1_dir = 'D:\DATA\Brats2018\Ver_2\Train\T1\5';
T1_save_dir = 'D:\DATA\Brats2018\Ver_2\Norm_slice\T1';
T1_files = dir(T1_dir); 

T1c_dir = 'D:\DATA\Brats2018\Ver_2\Train\T1c\5';
T1c_save_dir = 'D:\DATA\Brats2018\Ver_2\Norm_slice\T1c';
T1c_files = dir(T1c_dir); 

T2_dir = 'D:\DATA\Brats2018\Ver_2\Train\T2\5';
T2_save_dir = 'D:\DATA\Brats2018\Ver_2\Norm_slice\T2';
T2_files = dir(T2_dir); 

%%
a=0;b=0;
for i = 1:length(M_files)-2
    mask_name = fullfile(M_files(i+2).folder, M_files(i+2).name);
    mask = imread(mask_name);
    mask = mask(:);
    if all(mask==0)
        copyfile(fullfile(M_files(i+2).folder, M_files(i+2).name), M_save_dir);
        copyfile(fullfile(F_files(i+2).folder, F_files(i+2).name), F_save_dir);
        copyfile(fullfile(T1_files(i+2).folder, T1_files(i+2).name), T1_save_dir);
        copyfile(fullfile(T1c_files(i+2).folder, T1c_files(i+2).name), T1c_save_dir);
        copyfile(fullfile(T2_files(i+2).folder, T2_files(i+2).name), T2_save_dir);
    else
        b=b+1;
    end 
end