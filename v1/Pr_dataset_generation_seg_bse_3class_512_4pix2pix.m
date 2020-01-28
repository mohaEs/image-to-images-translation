
clc
clear
close all


PathHere=pwd;

path_orig='.\JSRT\JSRT\';
addpath(path_orig);
path_heart='.\Masks-1024\All Masks\heart\';
addpath(path_heart);
path_left='.\Masks-1024\All Masks\left lung\';
addpath(path_left);
path_right='.\Masks-1024\All Masks\right lung\';
addpath(path_right);
path_BSE='.\BSE_JSRT\BSE_JSRT\';

% orig
Dir=dir(fullfile(path_orig,'*.png'));
Pathname_orig=[PathHere '\512_Originals\'];
Pathname_seg=[PathHere '\512_Segments\'];
Pathname_bse=[PathHere '\512_BSEs\'];

mkdir('512_Originals')
mkdir('512_Segments')
mkdir('512_BSEs')

for i=1:length(Dir)
    [X,map]=imread(Dir(i).name);
    if ~isempty(map)
     Img = ind2rgb(X,map);
    end
    %imfinfo(Dir(i).name)
    % figure;imshow(Img,[]);
    
    % Intensity Normalization
    Img=double(Img);
    Img_resized=imresize(Img,[512 512]);
    Img_normalized=(Img_resized-min(Img_resized(:)))/(max(Img_resized(:))-min(Img_resized(:)));
    % figure;imshow(Img_normalized,[]);
    % min(Img_normalized(:))
    % max(Img_normalized(:))
    
    Img_normalized=uint8(Img_normalized*255);
    filename=[Pathname_orig Dir(i).name];
    imwrite(Img_normalized,filename);
    
    % Segments
    filename=[Dir(i).name(1:end-3) 'gif'];
    Img_Heart=imresize(imread([path_heart filename]),[512 512]);
    Img_Left=imresize(imread([path_left filename]),[512 512]);
    Img_Right=imresize(imread([path_right filename]),[512 512]);
    
    Mask(:,:,1)=Img_Heart;
    Mask(:,:,2)=Img_Left;
    Mask(:,:,3)=Img_Right;
    Mask4=Img_Right+Img_Left;
    Mask4=double(Mask4/255);
    
    % imshow(Mask);
    filename=[Pathname_seg Dir(i).name];
    imwrite(Mask,filename);
    
    %BSE
    filename=[Dir(i).name(1:end-3) 'png'];
    Img_BSE=imresize(double(imread([path_BSE filename])),[512 512]);
    Img_BSE_normalized=(Img_BSE-min(Img_BSE(:)))/(max(Img_BSE(:))-min(Img_BSE(:)));
    Img_BSE_normalized_masked=Mask4.*Img_BSE_normalized;
    Img_BSE_normalized=uint8(Img_BSE_normalized_masked*255);
    filename=[Pathname_bse Dir(i).name];
    imwrite(Img_BSE_normalized_masked,filename);   
           
end