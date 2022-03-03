# SR-with-SwinIR 
Using SwinIR to do the super-resolution task  
[Report!!](https://drive.google.com/file/d/1AgAtYsPpdXLypaov01X6h-s2sZJtzsQL/view?usp=sharing)
### Environment
python   : 3.8.11   
pytorch  : 1.9.0 + cu102

### Data prepare and preprocessing  
1. Download training dataset and put it into `./dataset/training_images/training_hr_images/`  
2. run `./dataset/training_images/downscale.py` to do the image preprocessing  
(Note: trimmed to multiple of 3 + make sure number of each dimension > 144)  


### Training
First, `cd SwinIR`  
To train SwinIR, run the following commands. You may need to modified the related .json file:  
(EX: classical SR, using `options/swinir/train_swinir_sr_classical.json` ),    
`dataroot_H`            : path for training set, high resolution image(groud truth),  
`dataroot_L`            : path for training set, low resolution image,  
`scale factor`          : setting scale for training (SR: 2,3,4,...),   
`dataloader_batch_size` : set the training batch size,    
and also,  `noisel level`, `JPEG level`, `G_optimizer_lr`, `G_scheduler_milestones`, etc. in the json file could be modified for different experiment scnario.  

And, modified the args below(you may directly modified it in `main_train_psnr.py`, or write it in the command )    
`--opt`           : path to related .json file,    
`--scale`         : setting scale for testing (SR: 2,3,4,...),    
`--folder_lq`     : path for val set, low resolution image,  
`--folder_gt`     : path for val set, high resolution image(groud truth),    
`--model_save_dir`: path for saving model(if model performance boost, do the model saving)  

checkpoint setting:   
(`checkpoint_test`, `checkpoint_save`, `checkpoint_print` in `options/swinir/train_swinir_sr_classical.json`)  
In original setting, the model would:  
print the training message for every 200 iterations,  
saving model for every 1000 iterations,  
testing with set5 in training process every 1000 iterations,  
Noted that: one iteration means one parameter update      

Train the model with line below:  
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json --dist True`

### Testing
For reproduing the submission:
1. Download the trained weight [(Link)](https://drive.google.com/file/d/1FuFTbK5vuE70G88livVOgCYLHIB-wI5l/view?usp=sharing)
2. Download the testing dataset. (Can put it in `https://github.com/skchen1993/VRDL_4_SR/tree/main/dataset/testing_images/testing_lr_images`)
3. run the command below

`CUDA_VISIBLE_DEVICES=0 python3 main_test_swinir_ensemble.py --task classical_sr --scale 3 --training_patch_size 48 --model_path {trained model path} --folder_lq {testing image path}`


## Citation
    @article{liang2021swinir,
        title={SwinIR: Image Restoration Using Swin Transformer},
        author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
        journal={arXiv preprint arXiv:2108.10257}, 
        year={2021}
    }

