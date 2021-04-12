bs=256
clusters=128
scheduler=cosine
weightdecay=7e-4
lambda=1.0
dataset=cifar10
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -m torch.distributed.launch \
--master_port 1233${GPU_ID} --nproc_per_node=1 train_cifar_npid_cld.py \
--batch-size ${bs} \
--lr 0.03 \
--epochs 200 \
--two-imgs \
--dataset ${dataset} \
--save-interval 50 \
--nce-k 0 \
--nce-t 0.1 \
--cld_t 0.2 \
--use-kmeans \
--clusters ${clusters} \
--num_iters 5 \
--Lambda ${lambda} \
--lr-scheduler ${scheduler} \
--warmup-epoch 5 \
--weight-decay ${weightdecay} \
--amp \
--opt-level O1 \
--num_workers 4 \
--recompute-memory \
--save-dir "checkpoint/${dataset}/NPID+CLD/resnet18/lr0.03-bs${bs}-clusters${clusters}-lambda${lambda}-${scheduler}-weightDecay${weightdecay}-fp16" \