lr=0.2
Lambda=0.25
cld_t=0.4
clusters=100
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_imagenet_moco_cld.py \
  -a resnet50 \
  --lr ${lr} \
  --workers 24 \
  --batch-size 512 \
  --moco-k 65536 \
  --Lambda 0.25 \
  --aug-plus --cos --mlp \
  --moco-t 0.2 \
  --cld-t ${cld_t} \
  --amp-opt-level O1 \
  --num-iters 5 \
  --clusters ${clusters} \
  --use-kmeans \
  --normlinear \
  --save-dir "output/imagenet/mocov2+cld/lr${lr}-Lambda${Lambda}-cld_t${cld_t}-clusters${clusters}-NormNLP-epochs200/" \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  DATA_DIR