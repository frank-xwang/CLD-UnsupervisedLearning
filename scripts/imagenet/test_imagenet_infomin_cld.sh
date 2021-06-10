CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_linear_infomin.py --method CLD \
  --aug_linear RA \
  --data_folder DATA_DIR \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --ckpt CHECKPOINT_PATH