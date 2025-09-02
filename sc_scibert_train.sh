#!/usr/bin/env bash
#SBATCH --job-name sbatch-t5-small
#SBATCH --array=0-0
#SBATCH --partition H100-SLT # A100-40GB A100-80GB H100-SLT H200
#SBATCH --nodes 1
#SBATCH --gpus 1 # 2
#SBATCH --cpus-per-gpu 2
#SBATCH --mem 128G # for CPU
#SBATCH --time 01-00:00   # 01-00:00 -> 1 day

N_PROC=1
MODEL_SIZE=small
SCORING=vanilla

# SRUN -START-
IMAGE=/netscratch/$USER/enroot/sc-v0.1.0.sqsh
MOUNTS="`pwd`/sc":/opt/sc,/netscratch/$USER:/netscratch/$USER,/netscratch/$USER:/ns,"`pwd`":/ws
MOUNTS=$MOUNTS,/home/$USER:/root

CONTAINER_ARGS="
    --container-image=$IMAGE \
    --container-mounts=$MOUNTS \
    --container-workdir=/ws
"
JOBNAME=exp-sc-${MODEL_SIZE}_$(date '+%Y%m%d-%H%M%S')
# SRUN -END-

# SCRIPT -START-
WANDB_PROJECT=sc-SciBERT
WANDB_TAGS=sc,${MODEL_SIZE},$SCORING
EXPORT_ARGS="'WANDB_PROJECT=$WANDB_PROJECT','WANDB_TAGS=$WANDB_TAGS','TOKENIZERS_PARALLELISM=false'"

SEED=1
RUN_NAME=sc_${MODEL_SIZE}_${SCORING}_$(date '+%Y%m%d-%H%M%S')
CONFIG_ARGS=(
    vocab_size=32128
    dropout_rate=0.1
    hidden_size=512
    feed_forward_proj=relu
    initializer_factor=1
    d_ff=2048
    d_kv=64
    layer_norm_epsilon=1e-6
    num_heads=8
    num_layers=6
)
CONFIG_ARGS=$(IFS=,; printf '%s' "${CONFIG_ARGS[*]}")
MODEL_ARGS="
    --model_type sc \
    --tokenizer_name sc-${MODEL_SIZE} \
    --config_overrides $CONFIG_ARGS
"

SEQ_LENGTH=512
DATA_ARGS="
    --dataset_name wmt/wmt14 \
    --dataset_config_name de-en \
    --overwrite_cache \
    --max_source_length $SEQ_LENGTH \
    --data_seed $SEED
"
DATALOADER_WORKER=$((4 * N_PROC))
TRAINING_ARGS="
    --do_train \
    --do_eval \
    --seed $SEED \
    --is_pretrain True \

    --mask_ratio 0.15 \
    --max_steps 500000 \
    --max_eval_samples 1000 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_steps 10000 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-08 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 128 \

    --logging_strategy steps \
    --logging_steps 25 \
    --eval_steps 250 \
    --eval_strategy steps \
    --save_strategy steps \
    --save_steps 1000 \
    --save_only_model \

    --fp16 \
    --torch_compile \
    --optim adamw_apex_fused \
    --dataloader_num_workers $DATALOADER_WORKER
"
RUN_ARGS="
    --run_name $RUN_NAME \
    --report_to wandb
"
# SCRIPT -END-

# run the training job
srun -K \
    --job-name=$JOBNAME \
    $CONTAINER_ARGS \
    --export=$EXPORT_ARGS \
    torchrun --standalone --nproc-per-node $N_PROC /ws/SciBERT_Classifier.py \
        $MODEL_ARGS \
        $DATA_ARGS \
        $TRAINING_ARGS \
        $RUN_ARGS \
        --output_dir /ns/models/$WANDB_PROJECT/$RUN_NAME
