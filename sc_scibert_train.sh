#!/usr/bin/env bash
#SBATCH --job-name sbatch-scibert
#SBATCH --array=0-4   # 5 domain：0,1,2,3,4
#SBATCH --partition A100-40GB # A100-40GB / A100-80GB / H100-SLT / H200 / V100-32GB
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem 48G
#SBATCH --time 3:00:00


# ---------------- Domain ----------------
DOMAINS=(cancer energy general neuroscience transport)
TRAIN_CSVS=(
    /ws/sc/src/sc/preprocessed/scilake_cancer_level2_hssc.csv
    /ws/sc/src/sc/preprocessed/scilake_energy_level2_hssc.csv
    /ws/sc/src/sc/preprocessed/scilake_general_level2_hssc.csv
    /ws/sc/src/sc/preprocessed/scilake_neuroscience_level2_hssc.csv
    /ws/sc/src/sc/preprocessed/scilake_transport_level2_hssc.csv
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
DOMAIN=${DOMAINS[$TASK_ID]}
TRAIN_CSV=${TRAIN_CSVS[$TASK_ID]}

echo ">>> Training domain-specific model for DOMAIN=$DOMAIN"
echo ">>> Using train_csv=$TRAIN_CSV"

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
WANDB_PROJECT=sc-SciBERT-$DOMAIN
WANDB_TAGS=sc,${MODEL_SIZE},$SCORING,$DOMAIN
EXPORT_ARGS="'WANDB_PROJECT=$WANDB_PROJECT','WANDB_TAGS=$WANDB_TAGS','TOKENIZERS_PARALLELISM=false','TRANSFORMERS_NO_TORCHVISION=1'"

SEED=42
RUN_NAME=sc_${MODEL_SIZE}_${SCORING}_${DOMAIN}_$(date '+%Y%m%d-%H%M%S')

CONFIG_ARGS=(
    dropout_rate=0.1
    layer_norm_epsilon=1e-12
)
CONFIG_ARGS=$(IFS=,; printf '%s' "${CONFIG_ARGS[*]}")

    # --model_type sc_hier \
# ---------------- MODEL_ARGS ----------------
MODEL_ARGS="
    --model_name allenai/scibert_scivocab_uncased \
    --tokenizer_name allenai/scibert_scivocab_uncased \
    --config_overrides $CONFIG_ARGS
"

# ---------------- DATA_ARGS -----------------
SEQ_LENGTH=256
DATA_ARGS="
    --train_csv $TRAIN_CSV \
    --max_length $SEQ_LENGTH
"

# ---------------- TRAINING_ARGS -------------
DATALOADER_WORKER=$((4 * N_PROC))
TRAINING_ARGS="
    --do_train \
    --do_eval \
    --seed $SEED \

    --epochs 5 \
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
    torchrun --standalone --nproc-per-node $N_PROC /ws/sc/src/sc/SciBERT_Classifier_train.py \
        $MODEL_ARGS \
        $DATA_ARGS \
        $TRAINING_ARGS \
        $RUN_ARGS \
        --output_dir /ns/models/$WANDB_PROJECT/$RUN_NAME
