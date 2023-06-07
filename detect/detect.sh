CUDA_VISIBLE_DEVICES=4 \
    CUDA_LAUNCH_BLOCKING=1 \
    OMP_NUM_THREADS=1 \
    python detect_fps.py \
    --checkpoint '/home/silee/workspace/ipiu_ssd/H-Fusion+Transformer+DN+CoordinateAttention_SoftLabel/checkpoint/Transformer_checkpointepoch_100.pth.tar' \
    --save_path '/home/silee/workspace/ipiu_ssd/H-Fusion+Transformer+DN+CoordinateAttention_SoftLabel/out/' \
    --save_name 'detect_fps.json'
# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 \
#     python detect.py \
#     --checkpoint '/home/silee/workspace/kroc/H-Fusion+Transformer+DN+CoordinateAttention_SoftLabel/checkpoint/Transformer_checkpointepoch_80.pth.tar' \
#     --save_path '/home/silee/workspace/kroc/H-Fusion+Transformer+DN+CoordinateAttention_SoftLabel/out/' \
#     --save_name 'epoch_80.json'

# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 \
#     python detect.py \
#     --checkpoint '/home/silee/workspace/kroc/H-Fusion+Transformer+DN+CoordinateAttention_SoftLabel/checkpoint/Transformer_checkpointepoch_85.pth.tar' \
#     --save_path '/home/silee/workspace/kroc/H-Fusion+Transformer+DN+CoordinateAttention_SoftLabel/out/' \
#     --save_name 'epoch_85.json'

# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 \
#     python detect.py \
#     --checkpoint '/home/silee/workspace/kroc/H-Fusion+Transformer+DN+CoordinateAttention_SoftLabel/checkpoint/Transformer_checkpointepoch_90.pth.tar' \
#     --save_path '/home/silee/workspace/kroc/H-Fusion+Transformer+DN+CoordinateAttention_SoftLabel/out/' \
#     --save_name 'epoch_90.json'

# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 \
#     python detect.py \
#     --checkpoint '/home/silee/workspace/kroc/H-Fusion+Transformer+DN+CoordinateAttention_SoftLabel/checkpoint/Transformer_checkpointepoch_95.pth.tar' \
#     --save_path '/home/silee/workspace/kroc/H-Fusion+Transformer+DN+CoordinateAttention_SoftLabel/out/' \
#     --save_name 'epoch_95.json'
