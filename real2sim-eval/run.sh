

LOG_PATH=/mnt/hdd/data/Robot_data/phystwin_data/
CUDA_VISIBLE_DEVICES=0 python experiments/eval_policy.py \
gs=rope \
env=xarm_gripper \
physics.ckpt_path=${LOG_PATH}/gs_scans/rope \
physics.case_name=rope_0001 \
policy.inference_cfg_path=policy/configs/inference/insert_rope.json \
policy.checkpoint_path=${LOG_PATH}/policy_checkpoints/act-rope-routing/checkpoints/007000