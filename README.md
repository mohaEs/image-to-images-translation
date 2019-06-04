# Image to images translation
# Multi-task pix2pix

This repository contains the codes and example images/folders for image-to-images translation. <br>
image-to-images translation is the extended version of image-to-image translation (pix2pix) which can learn and provide two out images. <br>

The codes are based on codes of pix2pix shared by https://github.com/affinelayer/pix2pix-tensorflow and is modified to address our desired goals.

This concept and code is used in the following research:
https://youtu.be/J8Uth26_7rQ


which you can find more information:


If you found this codes usefull, please cite the above paper.


## Main scripts
Main scripts are as follow (512x512 image size):

- pix2pix_0.py : original pix2pix 
- pix2pix_0_MT.py : Multi-task pix2pix 
- pix2pix_3_dG.py : Single-task pix2pix with dilation at stages 2 till 7 of the encoder of the generator network
- pix2pix_3_MTdG.py : Mutlit-task pix2pix with dilation at stages 2 till 7 of the encoder of the generator network

All the scripts can be called by the arguments: <br>
--mode   train/test  <br>
--output_dir   path/to/output  <br>
--input_dir   path/to/input  <br>
--checkpoint    path/to/trained/model  <br>
--max_epochs   2  <br>
--batch_size   2 <br>
--seed   1 <br>
--l1_weight   10 <br>

e.g. 
for training: <br>
> python ./Scripts/pix2pix_0.py --mode train  --output_dir ./Models/Model_orig_0_ST_T1  --input_dir ./512_Combined_Augmented/ST/T1/Train --which_direction AtoB --max_epochs 2 --batch_size 2 --seed 1 --l1_weight 10 

and for testing: <br>
> python ./Scripts/pix2pix_0.py --mode test --output_dir ./Results/Model_orig_0_ST_T1 --checkpoint ./Models/Model_orig_0_ST_T1 --input_dir ./512_Combined_Augmented/ST/T1/Test 

The codes are tested by tensorflow-gpu 1.12.0 and python 3.6.6


Now, we show more examples as follow.

## Data

For the singla tasks and multi tasks we should first prepare the dataset of combined images.<br>
e.g. for single task <br>
![Alt text](./512_Combined_Augmented/ST/T1/Val/JPCLN001.png?raw=true "Title") <br>
![Alt text](./512_Combined_Augmented/ST/T2/Val/JPCLN001.png?raw=true "Title")
and for multi-task: <br>
![Alt text](./512_Combined_Augmented/MT/Val/JPCLN001.png?raw=true "Title")

I put the MATLAB script Pr_dataset_generation_seg_bse_3class_512_4pix2pix.m which shows how to concatenate the images for tasks. <br>
I also shared the Pr_Combine_TrainValSplit.m for splitting the dataset to train/val/test.

For the mentioned research following datasets were used:
- Segmentation: https://www.isi.uu.nl/Research/Databases/SCR/
- Bone suppression: https://www.kaggle.com/hmchuong/xray-bone-shadow-supression

## Executing

Suppose we put the prepared data as arraneged in this repository in 512_Combined_Augmented folder.<br>
Do not forget tp set the values of batch size and epochs appropriatley.<br>
With following command shells, the models will be saved in Models folder and the results of tests will be saved on Results folder.<br>

### Single task 1 - pix2pix - train:
> python ./Scripts/pix2pix_0.py --mode train  --output_dir ./Models/Model_orig_0_ST_T1  --input_dir ./512_Combined_Augmented/ST/T1/Train --which_direction AtoB --max_epochs 2 --batch_size 2 --seed 1 --l1_weight 10
### Single task 1 - pix2pix - test:
> python ./Scripts/pix2pix_0.py --mode test --output_dir ./Results/Model_orig_0_ST_T1 --checkpoint ./Models/Model_orig_0_ST_T1 --input_dir ./512_Combined_Augmented/ST/T1/Test 
### Single task 2 - pix2pix - train:
> python ./Scripts/pix2pix_0.py --mode train  --output_dir ./Models/Model_orig_0_ST_T2  --input_dir ./512_Combined_Augmented/ST/T2/Train --which_direction AtoB --max_epochs 2 --batch_size 2 --seed 1 --l1_weight 10
### Single task 2 - pix2pix - test:
> python ./Scripts/pix2pix_0.py --mode test --output_dir ./Results/Model_orig_0_ST_T2 --checkpoint ./Models/Model_orig_0_ST_T2 --input_dir ./512_Combined_Augmented/ST/T2/Test 
### Single task 1 - pix2pix with dilation - train:
> python ./Scripts/pix2pix_3_dG.py --mode train  --output_dir ./Models/Model_orig_3_ST_T1  --input_dir ./512_Combined_Augmented/ST/T1/Train --which_direction AtoB --max_epochs 2 --batch_size 2 --seed 1 --l1_weight 10
### Single task 1 - pix2pix with dilation - test:
> python ./Scripts/pix2pix_3_dG.py --mode test --output_dir ./Results/Model_orig_3_ST_T1 --checkpoint ./Models/Model_orig_3_ST_T1 --input_dir ./512_Combined_Augmented/ST/T1/Test 
### Single task 2 - pix2pix with dilation - train:
> python ./Scripts/pix2pix_3_dG.py --mode train  --output_dir ./Models/Model_orig_3_ST_T2  --input_dir ./512_Combined_Augmented/ST/T2/Train --which_direction AtoB --max_epochs 2 --batch_size 2 --seed 1 --l1_weight 10
### Single task 2 - pix2pix with dilation - test:
> python ./Scripts/pix2pix_3_dG.py --mode test --output_dir ./Results/Model_orig_3_ST_T2 --checkpoint ./Models/Model_orig_3_ST_T2 --input_dir ./512_Combined_Augmented/ST/T2/Test 
### Mutli task pix2pix - train:
> python ./Scripts/pix2pix_0_MT.py --mode train  --output_dir ./Models/Model_orig_0_MT_A1  --input_dir ./512_Combined_Augmented/MT/Train --which_direction AtoB --max_epochs 2 --batch_size 2 --seed 1 --l1_weight 10
### Mutli task pix2pix - test:
> python ./Scripts/pix2pix_0_MT.py --mode test --output_dir ./Results/Model_orig_0_MT_A1 --checkpoint ./Models/Model_orig_0_MT_A1 --input_dir ./512_Combined_Augmented/MT/Test 
### Mutli task pix2pix with dilation - train:
> python ./Scripts/pix2pix_3_MTdG.py --mode train  --output_dir ./Models/Model_orig_3_MT_A1_1  --input_dir ./512_Combined_Augmented/MT/Train --which_direction AtoB --max_epochs 2 --batch_size 2 --seed 1 --l1_weight 10 
### Mutli task pix2pix with dilation- test:
> python ./Scripts/pix2pix_3_MTdG.py --mode test --output_dir ./Results/Model_orig_3_MT_A1_1 --checkpoint ./Models/Model_orig_3_MT_A1_1 --input_dir ./512_Combined_Augmented/MT/Test 


