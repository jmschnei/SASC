MOUNTS="`pwd`/sc":/opt/sc,/netscratch/$USER:/netscratch/$USER,/netscratch/$USER:/ns,"`pwd`":/ws
MOUNTS=$MOUNTS,/home/$USER:/root

srun \
    --container-image=/netscratch/$USER/enroot/sc-v0.1.0.sqsh \
    --container-mounts=$MOUNTS \
    --container-workdir=/ws \
    --partition=V100-16GB  \
    --time=02:00:00 \
    --immediate=300 \
    --mem=48GB \
    --pty \
    bash
