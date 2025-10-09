#!/usr/bin/env bash
#SBATCH --job-name sbatch-scibert
#SBATCH --array=0-0
#SBATCH --partition A100-40GB # A100-40GB / A100-80GB / H100-SLT / H200 / V100-32GB
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem 48G
#SBATCH --time 3:00:00

# ---------------- Base / SRUN ----------------
N_PROC=1
MODEL_SIZE=base
SCORING=vanilla

IMAGE=/netscratch/$USER/enroot/sc-v0.8.0.sqsh
MOUNTS="`pwd`/sc":/opt/sc,/netscratch/$USER:/netscratch/$USER,/netscratch/$USER:/ns,"`pwd`":/ws
MOUNTS=$MOUNTS,/home/$USER:/root

CONTAINER_ARGS="
    --container-image=$IMAGE \
    --container-mounts=$MOUNTS \
    --container-workdir=/ws
"
JOBNAME=exp-sc-${MODEL_SIZE}_$(date '+%Y%m%d-%H%M%S')

# ---------------- Script Vars ----------------
WANDB_PROJECT=sc-SciBERT
WANDB_TAGS=sc,${MODEL_SIZE},$SCORING
EXPORT_ARGS="'WANDB_PROJECT=$WANDB_PROJECT','WANDB_TAGS=$WANDB_TAGS','TOKENIZERS_PARALLELISM=false','TRANSFORMERS_NO_TORCHVISION=1'"

SEED=42
RUN_NAME=sc_${MODEL_SIZE}_${SCORING}_$(date '+%Y%m%d-%H%M%S')

CONFIG_ARGS=(
    dropout_rate=0.1
    layer_norm_epsilon=1e-12
)
CONFIG_ARGS=$(IFS=,; printf '%s' "${CONFIG_ARGS[*]}")

# ---------------- MODEL_ARGS ----------------
MODEL_ARGS="
    --model_type sc_hier \
    --model_name allenai/scibert_scivocab_uncased \
    --tokenizer_name allenai/scibert_scivocab_uncased \
    --config_overrides $CONFIG_ARGS
"

# ---------------- DATA_ARGS -----------------
SEQ_LENGTH=256
DATA_ARGS="
    --dataset_name nhop/academic-section-classification \
    --max_length $SEQ_LENGTH
"

# ---------------- TRAINING_ARGS -------------
DATALOADER_WORKER=$((4 * N_PROC))
TRAINING_ARGS="
    --do_train \
    --do_eval \
    --seed $SEED \

    --epochs 3 \
    --learning_rate 2e-5 \
    --batch_size 32 \

    --eval_strategy steps \
    --eval_steps 250 \
    --save_steps 1000 \

    --fp16 \
    --dataloader_num_workers $DATALOADER_WORKER
"

# ---------------- Prepare output dir --------
OUTDIR=/netscratch/$USER/models/$WANDB_PROJECT/$RUN_NAME
mkdir -p "$OUTDIR"


# ---------------- RUN_ARGS ------------------
RUN_ARGS="
    --run_name $RUN_NAME \
    --report_to wandb
"

# ---------------- Run -----------------------
srun -K \
    --job-name=$JOBNAME \
    $CONTAINER_ARGS \
    --export=$EXPORT_ARGS \
    torchrun --standalone --nproc-per-node $N_PROC /ws/sc/src/sc/HSSC_2level.py \
        $MODEL_ARGS \
        $DATA_ARGS \
        $TRAINING_ARGS \
        $RUN_ARGS \
        --output_dir /ns/models/$WANDB_PROJECT/$RUN_NAME
