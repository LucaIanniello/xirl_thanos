#!/bin/bash
#SBATCH --job-name=1Subtask_Xirl_Allo_Seed_42
#SBATCH --output=1Subtask_Xirl_Allo_Seed_42.log
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=3
#SBATCH --mem-per-cpu=4GB


source ~/miniconda3/etc/profile.d/conda.sh
conda activate xirl

# Remap CUDA_VISIBLE_DEVICES to match SLURM allocation
# export CUDA_VISIBLE_DEVICES=1,2
# echo "Remapped CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

PORT=$((29000 + SLURM_JOB_ID % 100))

# python test_embedding_6subtask.py --experiment_path /home/liannello/xirl/experiment_results/6Subtask/dataset=xmagical_mode=same_algo=resnet50_embodiment=gripper_uid=8465aafa-c7fe-4731-92b6-e2df14cea8dd

# python test_embedding_reds.py --experiment_path /home/liannello/xirl/experiment_results/6Subtask/dataset=xmagical_mode=same_algo=reds_embodiment=gripper_uid=Allo_6Subtask
python rl_xmagical_learned_reward_multi.py \
   --pretrained_path /home/liannello/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_EGO_6Subtask \
   --seeds 1 \
        --wandb \
    --port_number $PORT \
     --name_test 6Subtask_Xirl_Allo_Seed_42

# python compute_goal_embedding.py --experiment_path /home/liannello/xirl/experiment_results/6Subtask/dataset=xmagical_mode=same_algo=reds_embodiment=gripper
#MULTIGPU Pretraining
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT pretrain_same_embodiment_multi.py --embodiment gripper --algo dinov2 --wandb --unique_name Xirl_Dinov2_Allo_InvisibleRobot

#SingleGPU Pretraining
# python pretrain_xmagical_same_embodiment.py --embodiment gripper --algo resnet50 --unique_name --wandb

#STANDARD TRAINING
# python rl_xmagical_learned_reward.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_ALLO_6Subtasks --seeds 1 --wandb 

# MULTIGPU TRAINING
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/Egocentric/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_EGO_6Subtask --seeds 1 --wandb --name_test Egocentric_6SubtaskXirl_Curriculum_Normal

# # ALLOCENTRIC XIRL -> RUNNING
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_ALLO_6Subtasks --seeds 1 --wandb --name_test Allocentric_Xirl_Curriculum_20MTraining

# # ALLOCENTRIC DINOV2 -> RUNNING
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=dinov2_embodiment=gripper --seeds 1 --wandb --name_test Allocentric_Dinov2_6Subtasks_20MTraining

# # ALLOCENTRIC REDS -> RUMNING
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=reds_embodiment=gripper --seeds 1 --wandb --name_test Allocentric_Reds_6Subtasks_20MTraining

# # ALLOCENTRIC HOLDR -> RUNNING
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=holdr_embodiment=gripper --seeds 1 --wandb --name_test Allocentric_HolDR_6Subtasks_20MTraining

# # ALLOCENTRIC HOLDR_CONTRASTIVE -> RUNNING
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=holdr_embodiment=gripper_Contrastive --seeds 1 --wandb --name_test Allocentric_HolDR_Contrastive_6Subtasks_20MTraining

# # EGOCENTRIC XIRL -> RUNNING
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_EGO_6Subtask --seeds 1 --wandb --name_test Egocentric_Xirl_6Subtask20MTraining 

# # EGOCENTRIC DINOV2 -> RUNNING
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=dinov2_embodiment=gripper_EGO --seeds 1 --wandb --name_test Egocentric_Dinov2_6Subtask_20MTraining

# # EGOCENTRIC REDS -> RUNNING
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=reds_embodiment=gripper_EGO --seeds 1 --wandb --name_test Egocentric_Reds_6Subtask_20MTraining

# # EGOCENTRIC HOLDR
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=holdr_embodiment=gripper_EGO --seeds 1 --wandb --name_test Egocentric_HolDR_6Subtask_20MTraining

# # EGOCENTRIC HOLDR CONTRASTIVE
#torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_learned_reward_multi.py --pretrained_path /home/liannello/xirl/experiment_results/6Subtask/Egocentric_Pretrain/dataset=xmagical_mode=same_algo=holdr_embodiment=gripper_Contrastive_EGO --seeds 1 --wandb --name_test Egocentric_HolDR_Contrastive_6Subtask_20MTraining_Real

#VECTENV TRAINING
# torchrun --nproc_per_node=2 rl_xmagical_learned_reward_multi_vectEnv.py --pretrained_path /home/liannello/xirl/experiment_results/Egocentric/pretraining/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper=EGO_SubtaskXirl --seeds 1 --wandb --name_test 0.999G-MultiEnv-EGOEGO-5   

#ENV REWARD
# python rl_xmagical_env_reward_vectEnv.py --embodiment gripper --seeds 1 --wandb

# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT rl_xmagical_env_reward_multi_vectEnv.py --embodiment gripper --seeds 1 --name_test 20MillionMultiGPUENV
# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT  rl_xmagical_env_reward_multi.py --embodiment gripper --seeds 1 --name_test Allo-EnvReward-20Million-Video
# python generate_plot.py
# python test_embedding.py --experiment_path /home/liannello/xirl/experiment_results/Egocentric/pretraining/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper=EGO_SubtaskXirl

# python test_embedding_reds.py --experiment_path /home/liannello/xirl/experiment_results/Allocentric/pretraining/dataset=xmagical_mode=same_algo=reds_embodiment=gripper=ALLO_Reds

# torchrun --nproc_per_node=2 test_DDP.py
