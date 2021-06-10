CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_linear_infomin.py --method CLD \
  --aug_linear RA \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --data_folder DATA_DIR \
  --ckpt PRETRAINED_MODEL