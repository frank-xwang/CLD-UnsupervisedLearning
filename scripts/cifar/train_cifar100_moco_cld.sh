bs=128
clusters=100
scheduler=cosine
weightdecay=7e-4
lambda=0.25
cld_t=0.2
nce_t=0.07
nce_k=4096
bs_lr=0.03
dataset=cifar100
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -m torch.distributed.launch --master_port 1233${GPU_ID} --nproc_per_node=1 \
    train_cifar_moco_cld.py \
    --dataset ${dataset} \
    --num-workers 4 \
    --batch-size ${bs} \
    --nce-t ${nce_t} \
    --nce-k ${nce_k} \
    --base-learning-rate ${bs_lr} \
    --lr-scheduler ${scheduler} \
    --warmup-epoch 5 \
    --weight-decay ${weightdecay} \
    --cld_t ${cld_t} \
    --save-freq 100 \
    --three-imgs \
    --use-kmeans \
    --num-iters 5 \
    --Lambda ${lambda} \
    --normlinear \
    --save-dir "checkpoint/${dataset}/MoCo+CLD/resnet18/lr0.03-bs${bs}-cldT${cld_t}-nceT${nce_t}-clusters${clusters}-lambda${lambda}-${scheduler}-weightDecay${weightdecay}-fp16-add_erasing-kMeans-ncek${nce_k}-bslr${bs_lr}-normlinear" \