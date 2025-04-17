NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
OMP_NUM_THREADS=4 torchrun --nproc_per_node=$NUM_GPUS resnet50.py
