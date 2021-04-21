CUDA_ID=0,1,2,3,4,5,6,7
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_ID} \
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --workers 16 \
  --batch-size 256 \
  --save-dir SAVE_DIR \
  --pretrained PRETRAINED_MODEL \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  DATA_DIR