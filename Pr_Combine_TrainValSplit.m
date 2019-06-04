
clc
clear
close all


SourcePath='../../Dataset+Generated/512_Originals_Augmented/';
Task1Path='../../Dataset+Generated/512_Segments_Augmented/';
Task2Path='../../Dataset+Generated/512_BSEs_Augmented/';
SavePath='../../Dataset+Generated/512_Combined_Augmented/';


TrainValTestRatio=[0.75 0.1 0.15];

DirSource=dir(fullfile(SourcePath,'*.png'));
NumSamples=length(DirSource);


IndT=floor(TrainValTestRatio(1)*NumSamples);
IndV=IndT+floor(TrainValTestRatio(2)*NumSamples);

IndsRand=randperm(length(DirSource));

mkdir([SavePath 'MT/Train'])
mkdir([SavePath 'MT/Val'])
mkdir([SavePath 'MT/Test'])

mkdir([SavePath 'ST/T1/Train'])
mkdir([SavePath 'ST/T1/Val'])
mkdir([SavePath 'ST/T1/Test'])

mkdir([SavePath 'ST/T2/Train'])
mkdir([SavePath 'ST/T2/Val'])
mkdir([SavePath 'ST/T2/Test'])


for i=1:length(DirSource)
    Filename=DirSource(IndsRand(i)).name;
    
    Img=im2uint8(imread([SourcePath Filename]));
    if size(Img,3)==1
        Img = cat(3, Img, Img, Img);
    end
    %imshow(Img);
    
    ImgTask1=im2uint8(imread([Task1Path Filename]));
    if size(ImgTask1,3)==1
        ImgTask1 = cat(3, ImgTask1, ImgTask1, ImgTask1);
    end
    
    ImgTask2=im2uint8(imread([Task2Path Filename]));
    if size(ImgTask2,3)==1
        ImgTask2 = cat(3, ImgTask2, ImgTask2, ImgTask2);
    end
    %imshow(ImgTask2);
    ImgCombined_MT=cat(2,Img,ImgTask1,ImgTask2);
    ImgCombined_ST_1=cat(2,Img,ImgTask1);
    ImgCombined_ST_2=cat(2,Img,ImgTask2);

    %imshow(ImgCombined_MT);
    
    if i<=IndT
        imwrite(ImgCombined_MT,[SavePath 'MT/Train/' Filename]);
        imwrite(ImgCombined_ST_1,[SavePath 'ST/T1/Train/' Filename]);
        imwrite(ImgCombined_ST_2,[SavePath 'ST/T2/Train/' Filename]);
    elseif i<=IndV
        imwrite(ImgCombined_MT,[SavePath 'MT/Val/' Filename]);
        imwrite(ImgCombined_ST_1,[SavePath 'ST/T1/Val/' Filename]);
        imwrite(ImgCombined_ST_2,[SavePath 'ST/T2/Val/' Filename]);
    else
        imwrite(ImgCombined_MT,[SavePath 'MT/Test/' Filename]);
        imwrite(ImgCombined_ST_1,[SavePath 'ST/T1/Test/' Filename]);
        imwrite(ImgCombined_ST_2,[SavePath 'ST/T2/Test/' Filename]);
    end
    disp(Filename);
    
end

