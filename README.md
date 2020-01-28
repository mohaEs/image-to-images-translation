# image-to-images-translation
image to images translation - Multi task pix2pix

This repository contains the codes and example images/folders for image-to-images translation. <br>
image-to-images translation is the extended version of image-to-image translation (pix2pix) which can learn and provide two out images. <br>

The codes are based on codes of pix2pix shared by https://github.com/affinelayer/pix2pix-tensorflow and is modified to address our desired goals.

This concept and code is used in the following research:<br>
https://youtu.be/J8Uth26_7rQ <br>


which you can find more information:
https://arxiv.org/abs/1906.10089

If you found this codes usefull, please cite the above paper.

The codes are tested by python 3.5, 3.6 and tensorflow 1.12, 1.13 and 1.14.

This is the version v2, which makes the cross validation automatically. The v1 is also available and located in the folder v1.

## Main scripts
Main scripts located in scripts folder are as follow (512x512 image size):

- pix2pix_0orig_cv.py : original pix2pix 
- pix2pix_MT_cv.py : Multi-task pix2pix 
- pix2pix_dG_cv.py : Single-task pix2pix with dilation at stages 2 till 7 of the encoder of the generator network
- pix2pix_MTdG_cv.py : Mutlit-task pix2pix with dilation at stages 2 till 7 of the encoder of the generator network

All the scripts can be called by the arguments: <br>
--output_dir_all    &nbsp;  &nbsp;      path/to/dir/for/saving/outputs  <br>
--input_dir_all   &nbsp; &nbsp;  path/to/dir/images  <br>
--cv_info_dir &nbsp; &nbsp; path/to/dir/cv/setups  <br>
--task_No     &nbsp;  &nbsp; number indicating tasks: 1 and 2 for single tasks, 3 for multi-task <br>
--desired_l1_loss &nbsp;  &nbsp; desired l1 loss for early stopping <br>
--max_epochs  &nbsp; &nbsp;  maximum epochs  <br>
--batch_size  &nbsp; &nbsp; batch size <br>
--seed   &nbsp; &nbsp;   1 <br>
--l1_weight  &nbsp; &nbsp;  weight for l1 part in loss function <br>

e.g. 
for training with cross validation, single task, task 1, ... : <br>
> python ./Scripts/pix2pix_0orig_cv.py  --output_dir_all ./Outputs  --input_dir_all ./ImageData --cv_info_dir ./CV_info --task_No 1 --desired_l1_loss 0.05 --max_epochs 2000 --batch_size 2 --seed 1 --l1_weight 10  

and for multi-task: <br>
> python ./Scripts/pix2pix_MT_cv.py  --output_dir_all ./Outputs  --input_dir_all ./ImageData --cv_info_dir ./CV_info --task_No 3 --desired_l1_loss 0.05 --max_epochs 2000 --batch_size 2 --seed 1 --l1_weight 10 


The codes are tested by tensorflow-gpu 1.12.0 and python 3.6.6


Now, we show more examples as follow.

## Data

All the input images including, input, targets of tasks 1 and 2 should be placed in a folder and assigned to the input_dir_all arguments. Following image shows an example of the input folder:

![Alt text](./readme.jpg?raw=true "Title") <br>

For the mentioned research following datasets were used:
- Segmentation: https://www.isi.uu.nl/Research/Databases/SCR/
- Bone suppression: https://www.kaggle.com/hmchuong/xray-bone-shadow-supression


## cross validation setting

The names of image files which will be used for training/testing are deteremined by the train.txt and test.txt files which will be located in a folder and assigned to the cv_info_dir argument. Following image shows and example:

![Alt text](./readme_2.jpg?raw=true "Title") <br>

## Executing

Do not forget tp set the values of batch size and epochs appropriatley.<br>
With following command shells, the models  and results will be saved in output_dir_all folder.<br>
Notice that, if the folder contains previous saved models, the code will continue training. 


### Single task - task 1:
python ./Scripts/pix2pix_0orig_cv.py  --output_dir_all ./Outputs  --input_dir_all ./ImageData --cv_info_dir ./CV_info --task_No 1 --desired_l1_loss 0.05 --max_epochs 2000 --batch_size 2 --seed 1 --l1_weight 10 
### Single task - task 2:
python ./Scripts/pix2pix_0orig_cv.py  --output_dir_all ./Outputs  --input_dir_all ./ImageData --cv_info_dir ./CV_info --task_No 2 --desired_l1_loss 0.05 --max_epochs 2000 --batch_size 2 --seed 1 --l1_weight 10 
### Single task with dilation - task 1:
python ./Scripts/pix2pix_dG_cv.py  --output_dir_all ./Outputs  --input_dir_all ./ImageData --cv_info_dir ./CV_info --task_No 1 --desired_l1_loss 0.05 --max_epochs 2000 --batch_size 2 --seed 1 --l1_weight 10 
### Single task with dilation- task 2:
python ./Scripts/pix2pix_dG_cv.py  --output_dir_all ./Outputs  --input_dir_all ./ImageData --cv_info_dir ./CV_info --task_No 2 --desired_l1_loss 0.05 --max_epochs 2000 --batch_size 2 --seed 1 --l1_weight 10 
### multi task:
python ./Scripts/pix2pix_MT_cv.py  --output_dir_all ./Outputs  --input_dir_all ./ImageData --cv_info_dir ./CV_info --task_No 3 --desired_l1_loss 0.05 --max_epochs 2000 --batch_size 2 --seed 1 --l1_weight 10 
### multi task with dilation:
python ./Scripts/pix2pix_MTdG_cv.py  --output_dir_all ./Outputs  --input_dir_all ./ImageData --cv_info_dir ./CV_info --task_No 3 --desired_l1_loss 0.05 --max_epochs 2000 --batch_size 2 --seed 1 --l1_weight 10 


