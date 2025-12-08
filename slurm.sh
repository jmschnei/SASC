MOUNTS="`pwd`/sc":/opt/sc,/netscratch/$USER:/netscratch/$USER,/netscratch/$USER:/ns,"`pwd`":/ws
MOUNTS=$MOUNTS,/home/$USER:/root

    
srun \
    --container-image=/netscratch/$USER/enroot/sc-v0.8.0.sqsh \
    --container-mounts=$MOUNTS \
    --container-workdir=/ws \
    --partition=RTXA6000 \
    --gres=gpu:1 \
    --time=01:00:00 \
    --immediate=300 \
    --mem=48GB \
    --container-env=TRANSFORMERS_NO_TORCHVISION \
    --pty \
    bash
