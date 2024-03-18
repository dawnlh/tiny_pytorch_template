#!/bin/bash

# train
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "UNet" --mode "train" --data_dir "dataset/GOPRO_demo" 

# test
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "UNet" --mode "test" --data_dir "dataset/GOPRO_demo" --test_model "results/UNet/train/2024-03-18_19-48-06/weights/model_best.pth" 