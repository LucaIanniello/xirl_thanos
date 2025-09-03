# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default SAC config values."""

import ml_collections


def get_config():
  """Returns default config."""
  config = ml_collections.ConfigDict()

  # ================================================= #
  # Placeholders.
  # ================================================= #
  # These values will be filled at runtime once the gym.Env is loaded.
  obs_dim = ml_collections.FieldReference(None, field_type=int)
  action_dim = ml_collections.FieldReference(None, field_type=int)
  action_range = ml_collections.FieldReference(None, field_type=tuple)

  # ================================================= #
  # Main parameters.
  # ================================================= #
  config.save_dir = "/home/liannello/xirl/experiment_results/Egocentric/Results"

  # Set this to True to allow CUDA to find the best convolutional algorithm to
  # use for the given parameters. When False, cuDNN will deterministically
  # select the same algorithm at a possible cost in performance.
  config.cudnn_benchmark = True
  # Enforce CUDA convolution determinism. The algorithm itself might not be
  # deterministic so setting this to True ensures we make it repeatable.
  config.cudnn_deterministic = False

  # ================================================= #
  # Wrappers.
  # ================================================= #
  config.action_repeat = 1
  config.frame_stack = 3

  config.reward_wrapper = ml_collections.ConfigDict()
  config.reward_wrapper.pretrained_path = "/home/liannello/xirl/experiment_results/Egocentric/pretraining/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper=EGO_SubtaskXirl"
  # config.reward_wrapper.pretrained_path = None
  # Can be one of ['distance_to_goal', 'goal_classifier', holdr].
  config.reward_wrapper.type = "holdr"

  # Vector environment parameters for DDP
  config.num_envs_per_process = 3  # Number of parallel environments per DDP process

  # ================================================= #
  # Training parameters.
  # ================================================= #
  config.num_train_steps = 10_000_000
  config.replay_buffer_capacity = 1_000_000
  config.num_seed_steps = 30_000
  config.num_eval_episodes = 150
  config.eval_frequency = 50_000
  config.checkpoint_frequency = 300_000
  config.log_frequency = 20_000
  config.save_video = True

  # ================================================= #
  # SAC parameters.
  # ================================================= #
  config.sac = ml_collections.ConfigDict()

  config.sac.obs_dim = obs_dim
  config.sac.action_dim = action_dim
  config.sac.action_range = action_range
  config.sac.discount = 0.99
  config.sac.init_temperature = 0.3
  config.sac.alpha_lr = 1e-5
  config.sac.alpha_betas = [0.9, 0.999]
  config.sac.actor_lr = 1e-4
  config.sac.actor_betas = [0.9, 0.999]
  config.sac.actor_update_frequency = 1
  config.sac.critic_lr = 1e-5
  config.sac.critic_betas = [0.9, 0.999]
  config.sac.critic_tau = 0.005
  config.sac.critic_target_update_frequency = 2
  config.sac.batch_size = 2048
  config.sac.learnable_temperature = True

  # ================================================= #
  # Critic parameters.
  # ================================================= #
  config.sac.critic = ml_collections.ConfigDict()

  config.sac.critic.obs_dim = obs_dim
  config.sac.critic.action_dim = action_dim
  config.sac.critic.hidden_dim = 1024
  config.sac.critic.hidden_depth = 2

  # ================================================= #
  # Actor parameters.
  # ================================================= #
  config.sac.actor = ml_collections.ConfigDict()

  config.sac.actor.obs_dim = obs_dim
  config.sac.actor.action_dim = action_dim
  config.sac.actor.hidden_dim = 1024
  config.sac.actor.hidden_depth = 2
  config.sac.actor.log_std_bounds = [-5, 2]

  # ================================================= #

  return config
