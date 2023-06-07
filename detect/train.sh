CUDA_VISIBLE_DEVICES=7 \
OMP_NUM_THREADS=2 \
    python train.py \
    --model "H-Fusion+Transformer+DN+Coord+SoftLabel" \
    --workers 4 \
    --batch_size 32 \
    --epochs 150 \
    --print_freq 50 \
    --n_classes 2 \
    --lr 5e-4 \
    --weight_decay 5e-4 \
    --wandb_enable