# VRDL_4_SR
Using SwinIR to do the super-resolution task

## Environment
python   : 3.8.11   
pytorch  : 1.9.0 + cu102  

## Data preparation
1. Download the HR training image and put them into `/dataset/training_images/training_hr_trim_images/`    
2. run the `downscale.py` to do the image preprocessing  
(Trim the HR image to let the size be mutiple of 3, and then downscale for x3 LR training image)   
3. LR and trimmed HR training image would be placed in `/dataset/training_images/training_lr_images/`and`dataset/training_images/training_hr_trim_images/`  

## Model config setting
To train SwinIR, run the following commands. You may need to modified the related .json file:  
(EX: classical SR, using `options/swinir/train_swinir_sr_classical.json` ),    
`dataroot_H`            : path for training set, high resolution image(groud truth),  
`dataroot_L`            : path for training set, low resolution image,  
`scale factor`          : setting scale for training (SR: 2,3,4,...),   
`dataloader_batch_size` : set the training batch size,    
and also,  `noisel level`, `JPEG level`, `G_optimizer_lr`, `G_scheduler_milestones`, etc. in the json file could be modified for different experiment scnario.  

⭐⭐  
(0921)  
To do the GPU device selection, please use `CUDA_VISIBLE_DEVICES=0,3,....`in`train.sh `, or directly write it into command line like:  
 `CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json  --dist True `  
`gpu_ids` in `options/swinir/train_swinir_sr_classical.json` seems doesn't work....still finding the reason.    
⭐⭐  

And, modified the args below(you may directly modified it in `main_train_psnr.py`, or write it in the command )    
`--opt`           : path to related .json file,    
`--scale`         : setting scale for testing (SR: 2,3,4,...),    
`--folder_lq`     : path for testing set, low resolution image,  
`--folder_gt`     : path for testing set, high resolution image(groud truth),    
`--model_save_dir`: path for saving model(if model performance boost, do the model saving)    

Checkpoint setting:   
(`checkpoint_test`, `checkpoint_save`, `checkpoint_print` in `options/swinir/train_swinir_sr_classical.json`)  
Default setting:   
print the training message for every 200 iterations,  
saving model for every 3000 iterations,  
testing with validation set in training process every 3000 iterations,  
Noted that: one iteration means one parameter update  
 
Command below for training:     
```python
Classical Image SR
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json  --dist True
```

You can also train above models using `DataParallel` as follows, but it will be slower.
```python
# 001 Classical Image SR (middle size)
python main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json
```
