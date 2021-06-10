CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 main_imagenet_infomin_cld.py \
  --method CLD \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --amp --opt_level O1 \
  --cosine \
  --epochs 100 \
  -j 60 \
  --learning_rate 0.2 \
  --batch_size 512 \
  --dist-url 'tcp://127.0.0.1:23457' \
  --head 'normmlp' \
  --Lambda 0.25 \
  --cld-t 0.8 \
  --clusters 120 \
  --use-kmeans \
  --num-iters 5 \
  --model_path 'output/imagenet/infomin+cld/model/' \
  --tb_path 'output/imagenet/infomin+cld/tb/' \
  --data_folder DATA_DIR \